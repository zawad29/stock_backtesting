
from datetime import datetime, time
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.strategy import Strategy

class Backtest(BaseModel):
    """
    An event-driven backtesting engine with pydantic validation.
    """
    data: pd.DataFrame
    strategy: Strategy
    initial_capital: float = Field(100000.0, gt=0)
    shares_per_trade: int = Field(100, gt=0)
    session_start: time = Field(time(9, 30), description="Start of the trading session (e.g., 9:30 AM).")
    session_end: time = Field(time(16, 0), description="End of the trading session (e.g., 4:00 PM).")

    
    # Internal state - not part of the model's public interface
    signals: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)
    portfolio: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)
    trade_log: List[Dict[str, Any]] = Field(default_factory=list, repr=False)
    metrics: Dict[str, Any] = Field(default_factory=dict, repr=False)
    results_dir: str = Field("", repr=False)
    dow_stats: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)
    dom_stats: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)
    processed_trade_log: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)


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
        Runs the backtest using an iterative, event-driven approach with standard accounting.
        """
        position = 0  # -1 for short, 0 for flat, 1 for long
        cash = self.initial_capital
        holdings_value = 0  # Market value of current holdings

        for i in range(len(self.data)):
            current_timestamp = self.data.index[i]
            current_time = current_timestamp.time()
            current_price = self.data['Close'].iloc[i]

            # --- End of Session Logic: Force close any open position at or after session end ---
            if position != 0 and current_time >= self.session_end:
                if position == 1:  # Close long
                    cash += self.shares_per_trade * current_price
                    self.trade_log[-1].update({'Exit Price': current_price, 'Exit Time': current_timestamp})
                elif position == -1:  # Close short
                    cash -= self.shares_per_trade * current_price
                    self.trade_log[-1].update({'Exit Price': current_price, 'Exit Time': current_timestamp})
                position = 0
                holdings_value = 0
                self.portfolio.loc[current_timestamp, 'total'] = cash + holdings_value
                continue  # Move to the next bar without any further action today

            # --- Trading Logic: Only consider signals within the session ---
            is_in_session = self.session_start <= current_time < self.session_end
            
            if is_in_session:
                signal = self.signals['signal'].iloc[i]
                exit_signal = self.signals['exit_signal'].iloc[i]
                
                # Handle Exits first
                if position != 0 and exit_signal == 2:
                    if position == 1:  # Exit long
                        cash += self.shares_per_trade * current_price
                        self.trade_log[-1].update({'Exit Price': current_price, 'Exit Time': current_timestamp})
                        position = 0
                    elif position == -1:  # Exit short
                        cash -= self.shares_per_trade * current_price
                        self.trade_log[-1].update({'Exit Price': current_price, 'Exit Time': current_timestamp})
                        position = 0
                
                # Handle Entries if flat
                elif position == 0:
                    if signal == 1:  # Enter long
                        position = 1
                        cash -= self.shares_per_trade * current_price
                        self.trade_log.append(
                            {'Entry Time': current_timestamp, 'Entry Price': current_price, 'Type': 'Long'})
                    elif signal == -1:  # Enter short
                        position = -1
                        cash += self.shares_per_trade * current_price  # Receive cash from selling borrowed shares
                        self.trade_log.append(
                            {'Entry Time': current_timestamp, 'Entry Price': current_price, 'Type': 'Short'})

            # --- Update Portfolio Value for the current bar ---
            if position == 1:
                holdings_value = self.shares_per_trade * current_price
            elif position == -1:
                holdings_value = -self.shares_per_trade * current_price
            else:
                holdings_value = 0
            
            self.portfolio.loc[current_timestamp, 'total'] = cash + holdings_value

        # --- Force close any open position at the very end of the backtest ---
        if position != 0:
            final_price = self.data['Close'].iloc[-1]
            if position == 1:
                cash += self.shares_per_trade * final_price
            elif position == -1:
                cash -= self.shares_per_trade * final_price
            self.trade_log[-1].update({'Exit Price': final_price, 'Exit Time': self.data.index[-1]})
            self.portfolio.loc[self.data.index[-1], 'total'] = cash

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
        
        self.processed_trade_log = log.copy()
        
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
        if not self.processed_trade_log.empty:
            log_df_path = os.path.join(self.results_dir, 'day_trading_trade_log.csv')
            self.processed_trade_log.to_csv(log_df_path)
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

    def _plot_equity_curve(self, show_window: bool = False):
        fig, ax1 = plt.subplots(figsize=(15, 8), dpi=300)
        
        # Plot Equity Curve on Primary Y-axis
        ax1.plot(self.portfolio.index, self.portfolio['total'], lw=2., color='royalblue', label='Strategy Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)', color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')

        if not self.processed_trade_log.empty:
            trade_log_df = self.processed_trade_log
            # Plot trade markers on the equity curve
            long_trades = trade_log_df[trade_log_df['Type'] == 'Long']
            short_trades = trade_log_df[trade_log_df['Type'] == 'Short']
            ax1.plot(long_trades['Entry Time'], self.portfolio.total.loc[long_trades['Entry Time']],
                     '^', markersize=8, color='lime', label='Long Entry', alpha=0.9)
            ax1.plot(short_trades['Entry Time'], self.portfolio.total.loc[short_trades['Entry Time']],
                     'v', markersize=8, color='red', label='Short Entry', alpha=0.9)
            ax1.plot(trade_log_df['Exit Time'], self.portfolio.total.loc[trade_log_df['Exit Time']],
                     'x', markersize=8, color='black', label='Exit', alpha=0.9)
        
        # Create Secondary Y-axis for Daily PnL Bars
        ax2 = ax1.twinx()
        
        # --- CORRECTED Daily PnL Calculation ---
        if not self.processed_trade_log.empty:
            log_df = self.processed_trade_log
            daily_pnl = log_df.groupby(log_df['Exit Time'].dt.date)['PnL'].sum()
            daily_pnl.index = pd.to_datetime(daily_pnl.index)
            
            colors = ['limegreen' if pnl > 0 else 'salmon' for pnl in daily_pnl]
            ax2.bar(daily_pnl.index, daily_pnl.values, color=colors, width=0.8, alpha=0.6, label='Daily PnL')

        ax2.set_ylabel('Daily PnL ($)', color='dimgray')
        ax2.tick_params(axis='y', labelcolor='dimgray')
        ax2.axhline(0, color='grey', lw=0.5, linestyle='--')

        # Final Touches - Combine Legends
        ax1.set_title('Day Trading Strategy Performance with Daily PnL', fontsize=16)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right')

        metrics_text = '\n'.join([f'{key}: {value}' for key, value in self.metrics.items()])
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        
        fig.tight_layout()
        if show_window:
            plt.show()
        else:
            fig.savefig(os.path.join(self.results_dir, '1_equity_curve_with_daily_pnl.png'))
            plt.close(fig)

    def _plot_drawdown(self, show_window: bool = False):
        fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
        wealth_index = self.initial_capital * (1 + self.portfolio['returns'].fillna(0)).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        drawdowns.plot(ax=ax, kind='area', color='salmon', alpha=0.5)
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Portfolio Drawdown', fontsize=16)
        fig.tight_layout()
        if show_window:
            plt.show()
        else:
            fig.savefig(os.path.join(self.results_dir, '2_drawdown.png'))
            plt.close(fig)

    def _plot_dow_stats(self, show_window: bool = False):
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
            if show_window:
                plt.show()
            else:
                fig.savefig(os.path.join(self.results_dir, '3_day_of_week_stats.png'))
                plt.close(fig)

    def _plot_dom_stats(self, show_window: bool = False):
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
            if show_window:
                plt.show()
            else:
                fig.savefig(os.path.join(self.results_dir, '4_day_of_month_stats.png'))
                plt.close(fig)

    # --- Interactive Plotting Methods ---
    def _plot_equity_curve_interactive(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add Equity Curve
        fig.add_trace(go.Scatter(x=self.portfolio.index, y=self.portfolio['total'], mode='lines', name='Equity Curve'), secondary_y=False)
        
        if not self.processed_trade_log.empty:
            trade_log_df = self.processed_trade_log
            # Add Trade Markers
            long_trades = trade_log_df[trade_log_df['Type'] == 'Long']
            short_trades = trade_log_df[trade_log_df['Type'] == 'Short']
            fig.add_trace(go.Scatter(x=long_trades['Entry Time'], y=self.portfolio.total.loc[long_trades['Entry Time']],
                                     mode='markers', marker_symbol='triangle-up', marker_color='lime', marker_size=10, name='Long Entry'), secondary_y=False)
            fig.add_trace(go.Scatter(x=short_trades['Entry Time'], y=self.portfolio.total.loc[short_trades['Entry Time']],
                                     mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=10, name='Short Entry'), secondary_y=False)
            fig.add_trace(go.Scatter(x=trade_log_df['Exit Time'], y=self.portfolio.total.loc[trade_log_df['Exit Time']],
                                     mode='markers', marker_symbol='x', marker_color='black', marker_size=8, name='Exit'), secondary_y=False)
            
        # Add Daily PnL Bars (Corrected Logic)
        if not self.processed_trade_log.empty:
            log_df = self.processed_trade_log
            daily_pnl = log_df.groupby(log_df['Exit Time'].dt.date)['PnL'].sum()
            daily_pnl.index = pd.to_datetime(daily_pnl.index)

            colors = np.where(daily_pnl > 0, 'limegreen', 'salmon')
            fig.add_trace(go.Bar(x=daily_pnl.index, y=daily_pnl.values, name='Daily PnL', marker_color=colors, opacity=0.6), secondary_y=True)

        # Update Layout
        fig.update_layout(title_text='Interactive Equity Curve with Daily PnL', xaxis_rangeslider_visible=True)
        fig.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False)
        fig.update_yaxes(title_text="Daily PnL ($)", secondary_y=True)
        
        fig.write_html(os.path.join(self.results_dir, 'interactive_1_equity_curve_with_daily_pnl.html'))

    def _plot_drawdown_interactive(self):
        wealth_index = self.initial_capital * (1 + self.portfolio['returns'].fillna(0)).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns, fill='tozeroy', name='Drawdown', line_color='salmon'))
        fig.update_layout(title='Interactive Portfolio Drawdown', yaxis_title='Drawdown (%)')
        fig.write_html(os.path.join(self.results_dir, 'interactive_2_drawdown.html'))
        
    def _plot_dow_stats_interactive(self):
        if not self.dow_stats.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=self.dow_stats.index, y=self.dow_stats['Total PnL'], name='Total PnL', marker_color='royalblue'), secondary_y=False)
            fig.add_trace(go.Scatter(x=self.dow_stats.index, y=self.dow_stats['Win Rate (%)'], name='Win Rate (%)', marker_color='orange'), secondary_y=True)
            fig.update_layout(title_text='Interactive Performance by Day of Week')
            fig.update_yaxes(title_text="Total PnL ($)", secondary_y=False)
            fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0, 100])
            fig.write_html(os.path.join(self.results_dir, 'interactive_3_day_of_week_stats.html'))

    def _plot_dom_stats_interactive(self):
        if not self.dom_stats.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=self.dom_stats.index, y=self.dom_stats['Total PnL'], name='Total PnL', marker_color='seagreen'), secondary_y=False)
            fig.add_trace(go.Scatter(x=self.dom_stats.index, y=self.dom_stats['Win Rate (%)'], name='Win Rate (%)', marker_color='orange'), secondary_y=True)
            fig.update_layout(title_text='Interactive Performance by Day of Month', xaxis_title='Day of Month')
            fig.update_yaxes(title_text="Total PnL ($)", secondary_y=False)
            fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0, 100])
            fig.write_html(os.path.join(self.results_dir, 'interactive_4_day_of_month_stats.html'))
            
    def plot_to_file(self):
        """Generates static plots and saves them to files."""
        if 'Message' in self.metrics:
            print("Cannot generate static plots because no trades were made.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        
        self._plot_equity_curve(show_window=False)
        self._plot_drawdown(show_window=False)
        self._plot_dow_stats(show_window=False)
        self._plot_dom_stats(show_window=False)

        print(f"\nStatic plots saved to directory: '{self.results_dir}'")

    def plot_to_window(self):
        """Opens static plots in interactive GUI windows."""
        if 'Message' in self.metrics:
            print("Cannot generate plots because no trades were made.")
            return

        print("\n--- Opening Plots in Interactive Windows (close each window to continue) ---")
        plt.style.use('seaborn-v0_8-darkgrid')
        
        self._plot_equity_curve(show_window=True)
        self._plot_drawdown(show_window=True)
        self._plot_dow_stats(show_window=True)
        self._plot_dom_stats(show_window=True)

    def plot_interactive_to_html(self):
        """Generates interactive plots and saves them to HTML files."""
        if 'Message' in self.metrics:
            print("Cannot generate interactive plots because no trades were made.")
            return
        
        self._plot_equity_curve_interactive()
        self._plot_drawdown_interactive()
        self._plot_dow_stats_interactive()
        self._plot_dom_stats_interactive()

        print(f"\nInteractive plots saved to directory: '{self.results_dir}'")


