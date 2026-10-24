import pandas as pd


def add_features(df):
    df['ret_z'] = (df['ret'] - df['ret'].mean()) / df['ret'].std()
    df['vol_rank'] = df['vol'].rank(pct=True)
    df = df.fillna(df.median())
    return df
