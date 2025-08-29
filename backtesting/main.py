from backtest import Backtest
from strategies.intraday_rsi import IntradayRSI
import yfinance as yf


if __name__ == '__main__':
    
    ticker = 'AAPL'

    data = yf.download(ticker, period='60d', interval='5m')

    # Drop the multi-level columns
    data.columns = data.columns.droplevel(1)
    
    if data.empty:
        print(f"No intraday data downloaded for {ticker}. Please check the ticker or try again later.")
    else:
        try:
            print("--- Running Backtest ---")
            
            rsi_strategy = IntradayRSI()

            backtest = Backtest(data=data, strategy=rsi_strategy)
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
