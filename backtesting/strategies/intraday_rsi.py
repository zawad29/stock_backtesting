from strategies.strategy import Strategy
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IntradayRSI(Strategy, BaseModel):
    """
    An intraday RSI strategy for day trading, now with pydantic validation.
    - Buys when RSI is oversold.
    - Shorts when RSI is overbought.
    - Exits when RSI crosses the neutral level.
    """
    rsi_period: int = Field(14, gt=0, description="The period for RSI calculation.")
    oversold_threshold: int = Field(30, ge=0, lt=100, description="RSI threshold for oversold condition.")
    overbought_threshold: int = Field(70, ge=0, lt=100, description="RSI threshold for overbought condition.")
    neutral_threshold: int = Field(50, ge=0, lt=100, description="RSI neutral level for exiting trades.")

    class Config:
        arbitrary_types_allowed = True

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on the RSI indicator.
        """
        signals = pd.DataFrame(index=data.index)
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        signals['RSI'] = rsi

         # --- Generate Signals ---
        # 1 for Buy signal, -1 for Short signal, 0 for no action, 2 for Exit signal
        signals['signal'] = 0
        signals.loc[rsi < self.oversold_threshold, 'signal'] = 1
        signals.loc[rsi > self.overbought_threshold, 'signal'] = -1

        
        # Generate exit signal when RSI crosses neutral from either direction
        signals['exit_signal'] = np.where(
            ((signals['RSI'].shift(1) < self.neutral_threshold) & (signals['RSI'] >= self.neutral_threshold)) |
            ((signals['RSI'].shift(1) > self.neutral_threshold) & (signals['RSI'] <= self.neutral_threshold)),
            2, 0
        )
        
        return signals
    
