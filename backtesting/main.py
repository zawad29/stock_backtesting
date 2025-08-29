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
            rsi_strategy = IntradayRSI()
            backtest = Backtest(data=data, strategy=rsi_strategy)
            backtest.run()
            backtest.plot()
        except Exception as e:
            print(f"An error occurred: {e}")
