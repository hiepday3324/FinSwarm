import os
import datetime as dt
import pandas as pd
import numpy as np

def _normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).lower().strip() for col in df.columns]
    return df

def load_price_parquet(path: str) -> pd.DataFrame:
    """Load price data from a Parquet file.
    
    Required columns: symbol, date, close.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"
        ])
        
    df = pd.read_parquet(path)
    df = _normalize_df_columns(df)
    
    required = ["symbol", "date", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in price parquet {path}")
            
    # Normalize types
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["close"] = df["close"].astype(float)
    
    for opt_col in ["open", "high", "low", "adj_close", "volume"]:
        if opt_col not in df.columns:
            df[opt_col] = None
        else:
            df[opt_col] = df[opt_col].astype(object).where(df[opt_col].notna(), None)
            
    if "source" not in df.columns:
        df["source"] = "parquet"
    else:
        df["source"] = df["source"].astype(str)
        
    return df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]]

def load_news_parquet(path: str) -> pd.DataFrame:
    """Load news data from a Parquet file.
    
    Required columns: symbol, text.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "event_id", "symbol", "published_at", "available_at", "title", "text", "source", "url", "metadata"
        ])
        
    df = pd.read_parquet(path)
    df = _normalize_df_columns(df)
    
    required = ["symbol", "text"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in news parquet {path}")
            
    df["symbol"] = df["symbol"].astype(str)
    df["text"] = df["text"].astype(str)
    
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"])
    else:
        df["published_at"] = pd.Timestamp.now()
        
    if "available_at" in df.columns:
        df["available_at"] = pd.to_datetime(df["available_at"])
    else:
        df["available_at"] = df["published_at"]
        
    if "event_id" not in df.columns:
        # Generate hash based on symbol and text
        df["event_id"] = (df["symbol"] + "_" + df["published_at"].astype(str) + "_" + df["text"].str[:30]).apply(hash).astype(str)
    else:
        df["event_id"] = df["event_id"].astype(str)
        
    if "title" not in df.columns:
        df["title"] = None
        
    if "source" not in df.columns:
        df["source"] = "parquet"
    else:
        df["source"] = df["source"].astype(str)
        
    if "url" not in df.columns:
        df["url"] = None
        
    if "metadata" not in df.columns:
        df["metadata"] = [{}] * len(df)
    else:
        # ensure dict format
        df["metadata"] = df["metadata"].apply(lambda x: x if isinstance(x, dict) else {})
        
    return df[["event_id", "symbol", "published_at", "available_at", "title", "text", "source", "url", "metadata"]]

def load_filings_parquet(path: str) -> pd.DataFrame:
    """Load SEC filings / financial reports from a Parquet file.
    
    Required columns: symbol, text.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "filing_id", "symbol", "filing_type", "filing_date", "accepted_at", "available_at", "text", "source", "url", "metadata"
        ])
        
    df = pd.read_parquet(path)
    df = _normalize_df_columns(df)
    
    required = ["symbol", "text"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in filings parquet {path}")
            
    df["symbol"] = df["symbol"].astype(str)
    df["text"] = df["text"].astype(str)
    
    if "filing_type" not in df.columns:
        df["filing_type"] = "10-Q"
        
    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    else:
        df["filing_date"] = dt.date.today()
        
    if "accepted_at" in df.columns:
        df["accepted_at"] = pd.to_datetime(df["accepted_at"])
    else:
        df["accepted_at"] = None
        
    if "available_at" in df.columns:
        df["available_at"] = pd.to_datetime(df["available_at"])
    else:
        df["available_at"] = pd.to_datetime(df["filing_date"])
        
    if "filing_id" not in df.columns:
        df["filing_id"] = (df["symbol"] + "_" + df["filing_date"].astype(str) + "_" + df["text"].str[:30]).apply(hash).astype(str)
    else:
        df["filing_id"] = df["filing_id"].astype(str)
        
    if "source" not in df.columns:
        df["source"] = "parquet"
    else:
        df["source"] = df["source"].astype(str)
        
    if "url" not in df.columns:
        df["url"] = None
        
    if "metadata" not in df.columns:
        df["metadata"] = [{}] * len(df)
    else:
        df["metadata"] = df["metadata"].apply(lambda x: x if isinstance(x, dict) else {})
        
    return df[["filing_id", "symbol", "filing_type", "filing_date", "accepted_at", "available_at", "text", "source", "url", "metadata"]]

def load_market_parquets(base_dir: str) -> dict[str, pd.DataFrame]:
    """Load all prices, news, and filings from parquet files in a directory."""
    prices_path = os.path.join(base_dir, "prices.parquet")
    news_path = os.path.join(base_dir, "news.parquet")
    filings_path = os.path.join(base_dir, "filings.parquet")
    
    return {
        "prices": load_price_parquet(prices_path),
        "news": load_news_parquet(news_path),
        "filings": load_filings_parquet(filings_path),
    }
