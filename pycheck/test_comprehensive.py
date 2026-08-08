"""
Comprehensive test file for PyCheck — exercises ALL rules.
Every line below contains an intentional bug that PyCheck should catch.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def build_features(df):
    # T001: shift(-1) future access
    df['future_price'] = df['close'].shift(-1)
    
    # T003: rolling before split
    df['sma'] = df['close'].rolling(14).mean()
    
    # T004: expanding before split
    df['cummax'] = df['close'].expanding().max()
    
    # T005: pct_change negative
    df['future_ret'] = df['close'].pct_change(-1)
    
    # T007: backfill leaks future
    df['filled'] = df['close'].bfill()
    df['filled2'] = df['close'].fillna(method='bfill')
    
    # T008: non-causal interpolation
    df['interp'] = df['close'].interpolate(method='cubic')
    
    # T009: centered rolling
    df['centered'] = df['close'].rolling(10, center=True).mean()
    
    # T013: resample before split
    df_monthly = df.resample('M').mean()
    
    # T014: diff negative
    df['future_diff'] = df['close'].diff(-1)
    
    # C001: correlation for decisions
    corr = df['close'].corr()
    
    # C003: post-treatment variable
    df['after_treatment'] = df['outcome'] * 2
    
    # C007: corrwith for feature selection
    best_features = df.corrwith(df['target'])
    
    return df

def train_model(df):
    # T002: train_test_split on time series
    X_train, X_test, y_train, y_test = train_test_split(df[['sma']], df['target'])
    
    # T010: fit_transform on full data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['sma', 'close']])
    
    # T012: KFold on time series
    cv = KFold(n_splits=5)
    
    # U004: single model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # U001: predict without uncertainty
    predictions = model.predict(X_test)
    
    # U005: score on training data
    train_score = model.score(X_train, y_train)
    
    # C004: selection on outcome
    df = df[df['profit'] > 0]
    
    # C005: survivorship via dropna
    df = df.dropna()
    
    return model

def backtest(model, df):
    # T006: future index in loop
    for i in range(len(df) - 1):
        current = df['close'].iloc[i]
        future = df['close'].iloc[i + 1]
        
        # U002: hardcoded threshold
        if prediction > 0.5:
            print("buy")
    
    # U006: predict in production loop
    for i in range(100):
        pred = model.predict(df.iloc[[i]])
    
    # C006: p-hacking
    if p_value < 0.05:
        print("significant")

def bad_error_handling():
    # D002: silent exception swallow
    try:
        data = pd.read_csv("data.csv")
    except:
        pass

# C002: target in feature list
features = df[['sma', 'target', 'volume']]
