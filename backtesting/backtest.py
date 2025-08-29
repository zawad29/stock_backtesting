
from datetime import datetime
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from strategies.strategy import Strategy

class Backtest(BaseModel):
    """
    An event-driven backtesting engine with pydantic validation.
    """
    data: pd.DataFrame
    strategy: Strategy
    initial_capital: float = Field(100000.0, gt=0)
    shares_per_trade: int = Field(100, gt=0)
    
    # Internal state - not part of the model's public interface
    signals: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)
    portfolio: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)
    trade_log: List[Dict[str, Any]] = Field(default_factory=list, repr=False)
    metrics: Dict[str, Any] = Field(default_factory=dict, repr=False)
    results_dir: str = Field("", repr=False)
    dow_stats: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)
    dom_stats: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)


    class Config:
        arbitrary_types_allowed = True # Allow pandas DataFrame

    def model_post_init(self, __context: Any) -> None:
        """Initialize internal state after pydantic model validation."""
        self.signals = self.strategy.generate_signals(self.data)
        self.portfolio = pd.DataFrame(index=self.data.index)
        # Create a timestamped directory for the results
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.results_dir = os.path.join('result', timestamp)
        os.makedirs(self.results_dir, exist_ok=True)


    def run(self):
        """
        Runs the backtest using an iterative, event-driven approach.
        """
        position = 0  # -1 for short, 0 for flat, 1 for long
        cash = self.initial_capital
        holdings = 0
        
        for i in range(len(self.data)):
            # --- End of Day Logic ---
            is_end_of_day = i > 0 and self.data.index[i].date() != self.data.index[i-1].date()
            if is_end_of_day and position != 0:
                close_price = self.data['Close'].iloc[i-1]
                if position == 1:
                    cash += self.shares_per_trade * close_price
                elif position == -1:
                    cash += self.shares_per_trade * (2 * self.trade_log[-1]['Entry Price'] - close_price)
                self.trade_log[-1].update({'Exit Price': close_price, 'Exit Time': self.data.index[i-1]})
                position = 0

            # --- Trading Logic ---
            signal = self.signals['signal'].iloc[i]
            exit_signal = self.signals['exit_signal'].iloc[i]
            current_price = self.data['Close'].iloc[i]

            if position != 0 and exit_signal == 2:
                if position == 1:
                    cash += self.shares_per_trade * current_price
                elif position == -1:
                    cash += self.shares_per_trade * (2 * self.trade_log[-1]['Entry Price'] - current_price)
                self.trade_log[-1].update({'Exit Price': current_price, 'Exit Time': self.data.index[i]})
                position = 0
            
            elif position == 0:
                if signal == 1:
                    position = 1
                    cash -= self.shares_per_trade * current_price
                    self.trade_log.append({'Entry Time': self.data.index[i], 'Entry Price': current_price, 'Type': 'Long'})
                elif signal == -1:
                    position = -1
                    self.trade_log.append({'Entry Time': self.data.index[i], 'Entry Price': current_price, 'Type': 'Short'})

            # --- Update Portfolio Value ---
            if position == 1:
                holdings = self.shares_per_trade * current_price
            elif position == -1:
                entry_price = self.trade_log[-1]['Entry Price']
                holdings = self.shares_per_trade * (2 * entry_price - current_price)
            else:
                holdings = 0
            
            self.portfolio.loc[self.data.index[i], 'total'] = cash + holdings

        self.portfolio['returns'] = self.portfolio['total'].pct_change()
        self._calculate_metrics()
        self.generate_csv_report()

    def _calculate_metrics(self):
        log = pd.DataFrame(self.trade_log).dropna()
        if log.empty:
            self.metrics = { 'Message': "No trades were executed." }
            print(self.metrics['Message'])
            return

        # Ensure datetime type for analysis
        log['Exit Time'] = pd.to_datetime(log['Exit Time'])
        
        log['PnL'] = np.where(log['Type'] == 'Long', 
                              (log['Exit Price'] - log['Entry Price']) * self.shares_per_trade,
                              (log['Entry Price'] - log['Exit Price']) * self.shares_per_trade)
        
        total_return = (self.portfolio['total'].iloc[-1] / self.initial_capital) - 1
        returns_std = self.portfolio['returns'].std()
        sharpe_ratio = np.sqrt(252 * 78) * (self.portfolio['returns'].mean() / returns_std) if returns_std != 0 else 0
        
        wealth_index = self.initial_capital * (1 + self.portfolio['returns']).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()

        win_rate = (log['PnL'] > 0).sum() / len(log) * 100 if len(log) > 0 else 0
        total_profit = log[log['PnL'] > 0]['PnL'].sum()
        total_loss = abs(log[log['PnL'] < 0]['PnL'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        self.metrics = {
            'Cumulative Return': f"{total_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Maximum Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2f}%",
            'Profit Factor': f"{profit_factor:.2f}",
            'Total Trades': len(log)
        }
        
        # --- Detailed Stats Calculation ---
        log['day_of_week'] = log['Exit Time'].dt.day_name()
        log['day_of_month'] = log['Exit Time'].dt.day

        # Day of Week Analysis
        dow_stats = log.groupby('day_of_week')['PnL'].agg(['count', lambda pnl: (pnl > 0).sum(), 'sum'])
        dow_stats.columns = ['Total Trades', 'Winning Trades', 'Total PnL']
        dow_stats['Win Rate (%)'] = (dow_stats['Winning Trades'] / dow_stats['Total Trades']) * 100
        dow_stats = dow_stats.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']).dropna(how='all')
        self.dow_stats = dow_stats.round(2)

        # Day of Month Analysis
        dom_stats = log.groupby('day_of_month')['PnL'].agg(['count', lambda pnl: (pnl > 0).sum(), 'sum'])
        dom_stats.columns = ['Total Trades', 'Winning Trades', 'Total PnL']
        dom_stats['Win Rate (%)'] = (dom_stats['Winning Trades'] / dom_stats['Total Trades']) * 100
        self.dom_stats = dom_stats.round(2)

        print("Backtesting Metrics:")
        for key, value in self.metrics.items():
            print(f"- {key}: {value}")
        
        print("\n--- Performance by Day of Week ---")
        print(self.dow_stats.to_string())

        print("\n--- Performance by Day of Month ---")
        print(self.dom_stats.to_string())

    def generate_csv_report(self):
        portfolio_path = os.path.join(self.results_dir, 'day_trading_backtest_results.csv')
        self.portfolio.to_csv(portfolio_path)
        
        log_path_info = ""
        if self.trade_log:
            log_df_path = os.path.join(self.results_dir, 'day_trading_trade_log.csv')
            pd.DataFrame(self.trade_log).to_csv(log_df_path)
            log_path_info = f" and '{log_df_path}'"

        print(f"\nReports saved to '{portfolio_path}'{log_path_info}")

        if not self.dow_stats.empty:
            dow_stats_path = os.path.join(self.results_dir, 'day_of_week_stats.csv')
            self.dow_stats.to_csv(dow_stats_path)
            print(f"Day of week stats saved to '{dow_stats_path}'")
        
        if not self.dom_stats.empty:
            dom_stats_path = os.path.join(self.results_dir, 'day_of_month_stats.csv')
            self.dom_stats.to_csv(dom_stats_path)
            print(f"Day of month stats saved to '{dom_stats_path}'")

    def _plot_equity_curve(self):
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        self.portfolio['total'].plot(ax=ax, lw=2., label='Strategy Equity Curve')
        trade_log_df = pd.DataFrame(self.trade_log).dropna()
        if not trade_log_df.empty:
            long_trades = trade_log_df[trade_log_df['Type'] == 'Long']
            short_trades = trade_log_df[trade_log_df['Type'] == 'Short']
            ax.plot(long_trades['Entry Time'], self.portfolio.total.loc[long_trades['Entry Time']],
                     '^', markersize=8, color='lime', label='Long Entry', alpha=0.8)
            ax.plot(short_trades['Entry Time'], self.portfolio.total.loc[short_trades['Entry Time']],
                     'v', markersize=8, color='red', label='Short Entry', alpha=0.8)
            ax.plot(trade_log_df['Exit Time'], self.portfolio.total.loc[trade_log_df['Exit Time']],
                     'x', markersize=8, color='black', label='Exit', alpha=0.8)
        ax.set_title('Day Trading Strategy Performance', fontsize=16)
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        metrics_text = '\n'.join([f'{key}: {value}' for key, value in self.metrics.items()])
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, '1_equity_curve.png'))
        plt.close(fig)

    def _plot_drawdown(self):
        fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
        wealth_index = self.initial_capital * (1 + self.portfolio['returns'].fillna(0)).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        drawdowns.plot(ax=ax, kind='area', color='salmon', alpha=0.5)
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Portfolio Drawdown', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, '2_drawdown.png'))
        plt.close(fig)

    def _plot_dow_stats(self):
        if not self.dow_stats.empty:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            self.dow_stats['Total PnL'].plot(kind='bar', ax=ax, color='royalblue', alpha=0.7)
            ax.set_ylabel('Total PnL ($)')
            ax.set_title('Performance by Day of Week', fontsize=16)
            axb = ax.twinx()
            self.dow_stats['Win Rate (%)'].plot(kind='line', ax=axb, color='orange', marker='o', lw=2)
            axb.set_ylabel('Win Rate (%)')
            axb.set_ylim(0, 100)
            fig.tight_layout()
            fig.savefig(os.path.join(self.results_dir, '3_day_of_week_stats.png'))
            plt.close(fig)

    def _plot_dom_stats(self):
        if not self.dom_stats.empty:
            fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
            self.dom_stats['Total PnL'].plot(kind='bar', ax=ax, color='seagreen', alpha=0.7)
            ax.set_ylabel('Total PnL ($)')
            ax.set_title('Performance by Day of Month', fontsize=16)
            ax.set_xlabel('Day of Month')
            axb = ax.twinx()
            self.dom_stats['Win Rate (%)'].plot(kind='line', ax=axb, color='orange', marker='o', lw=2)
            axb.set_ylabel('Win Rate (%)')
            axb.set_ylim(0, 100)
            fig.tight_layout()
            fig.savefig(os.path.join(self.results_dir, '4_day_of_month_stats.png'))
            plt.close(fig)
            
    def plot(self):
        if 'Message' in self.metrics:
            print("Cannot generate plot because no trades were made.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        
        self._plot_equity_curve()
        self._plot_drawdown()
        self._plot_dow_stats()
        self._plot_dom_stats()

        print(f"\nPlots saved to directory: '{self.results_dir}'")
