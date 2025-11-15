import numpy as np
import pandas as pd
from typing import Dict, Tuple

def calculate_garman_klass_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    term = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    vol = np.sqrt(term.rolling(window=window).mean() * 252.0)
    return vol

def calculate_parkinson_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
    log_hl = np.log(df["high"] / df["low"]) ** 2
    vol = np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl.rolling(window=window).mean() * 252.0)
    return vol

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    z_score = (series - mid) / (std + 1e-9)
    return upper, lower, mid, z_score

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    features["gk_vol"] = calculate_garman_klass_volatility(df).bfill().fillna(0.0)
    features["rsi"] = (calculate_rsi(df["close"]) / 100.0 - 0.5).fillna(0.0)
    _, _, macd_hist = calculate_macd(df["close"])
    features["macd_hist"] = (macd_hist / df["close"]).fillna(0.0)
    _, _, _, bz = calculate_bollinger_bands(df["close"])
    features["bollinger_z"] = bz.fillna(0.0)
    features["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean() - 1.0).fillna(0.0)
    return features
