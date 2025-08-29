from datetime import time
from backtest import Backtest
from strategies.intraday_rsi import IntradayRSI
import yfinance as yf



if __name__ == '__main__':
    # --- Main Configuration ---
    # Easily change your backtest parameters here
    config = {
        "ticker": "AAPL",
        "period": "60d",  # e.g., "60d" for 60 days, "1y" for 1 year
        "interval": "15m", # e.g., "1m", "5m", "15m", "1h"
        "session_start_time": "09:30",
        "session_end_time": "15:55"
    }

    print(f"--- Downloading Data for {config['ticker']} ---")
    data = yf.download(
        tickers=config['ticker'],
        period=config['period'],
        interval=config['interval']
    )

    # Drop the multi-level columns
    data.columns = data.columns.droplevel(1)
    
    if data.empty:
        print(f"No intraday data downloaded for {config['ticker']}. Please check the ticker or try again later.")
    else:
        try:
            print("\n--- Running Backtest ---")
            rsi_strategy = IntradayRSI()

            backtest = Backtest(
                data=data, 
                strategy=rsi_strategy,
                session_start=time.fromisoformat(config['session_start_time']),
                session_end=time.fromisoformat(config['session_end_time'])
            )

            backtest.run()
            
            # --- Choose Your Plotting Method ---
            # 1. Save static plots to files
            print("\n--- Generating Static Plots (PNG) ---")
            backtest.plot_to_file()

            # 2. Save interactive plots to HTML files
            print("\n--- Generating Interactive Plots (HTML) ---")
            backtest.plot_interactive_to_html()

            # 3. Open plots in new windows (uncomment the line below to use)
            # backtest.plot_to_window()

        except Exception as e:
            print(f"An error occurred: {e}")

            
