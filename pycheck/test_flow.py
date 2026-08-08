"""Test file for data flow taint propagation — PyCheck should detect
that tainted data from shift(-1) flows through assignments into model.fit()"""
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("prices.csv")

# Step 1: Create tainted data
future_price = df['close'].shift(-1)

# Step 2: Taint propagates through arithmetic
signal = future_price - df['close']

# Step 3: Taint propagates through assignment
features = signal * 100

# Step 4: Tainted data reaches a training sink
model = LinearRegression()
model.fit(df[['volume']], features)  # F001: tainted data in training!

# Also test: predict with tainted features
predictions = model.predict(df[['volume']])
