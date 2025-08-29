from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    Ensures that any strategy inheriting from it must implement the generate_signals method.
    """
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals for the given data.
        
        Args:
            data: A pandas DataFrame with historical market data (OHLCV).

        Returns:
            A pandas DataFrame with signal information.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
