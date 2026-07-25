"""
stock_predictor.py — A "typical" ML stock prediction pipeline.
This is the kind of code that loses companies millions.
It looks correct. It runs without errors. But it has 6 hidden bugs.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
#  Load and prepare data
# ─────────────────────────────────────────
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
prices = np.cumsum(np.random.randn(1000)) + 100
volume = np.random.randint(1000, 10000, 1000).astype(float)

df = pd.DataFrame({
    'date': dates,
    'close': prices,
    'volume': volume
})

# ─────────────────────────────────────────
#  BUG 1: TEMPORAL LEAK — shift(-1) uses FUTURE prices as the target
#  The model is trained to predict tomorrow's price using today's features,
#  but shift(-1) peeks into the future. In production, this data doesn't exist.
# ─────────────────────────────────────────
df['target'] = df['close'].shift(-1)

# ─────────────────────────────────────────
#  BUG 2: TEMPORAL LEAK — Features computed on FULL dataset before split
#  The rolling mean includes test data in its computation.
#  This means the model has seen the test set during training.
# ─────────────────────────────────────────
df['rolling_mean_20'] = df['close'].rolling(20).mean()
df['rolling_std_20'] = df['close'].rolling(20).std()
df['momentum'] = df['close'].pct_change(5)

# ─────────────────────────────────────────
#  BUG 3: TEMPORAL LEAK — Future data in feature engineering
#  shift(-5) creates a feature from 5 days IN THE FUTURE
# ─────────────────────────────────────────
df['future_momentum'] = df['close'].shift(-5) / df['close'] - 1

df = df.dropna()

features = ['close', 'volume', 'rolling_mean_20', 'rolling_std_20', 'momentum', 'future_momentum']
X = df[features].values
y = df['target'].values

# ─────────────────────────────────────────
#  BUG 4: TEMPORAL LEAK — Random train/test split on time series
#  train_test_split shuffles the data, so the model trains on
#  future data and tests on past data. For time series, you must
#  use a temporal split (everything before date X = train, after = test).
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────
#  Train the model
# ─────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ─────────────────────────────────────────
#  BUG 5: UNGUARDED UNCERTAINTY — Prediction used without confidence check
#  The model returns a point estimate with no confidence interval.
#  A prediction of $150 with R²=0.99 is very different from
#  $150 with R²=0.1, but the code treats them identically.
# ─────────────────────────────────────────
predictions = model.predict(X_test)

# ─────────────────────────────────────────
#  BUG 6: CAUSAL CONFUSION — Correlation treated as causation
#  "Volume correlates with price changes" does NOT mean
#  "increasing volume CAUSES price to go up."
#  But the model uses correlation to make trading decisions.
# ─────────────────────────────────────────
correlation = df[['close', 'volume']].corr()
if correlation.iloc[0, 1] > 0.5:
    trading_signal = "BUY"  # Assumes correlation = causation
else:
    trading_signal = "SELL"

# Generate trading signals from raw predictions
for i, pred in enumerate(predictions[:10]):
    action = "BUY" if pred > y_test[i] else "SELL"
    print(f"Day {i}: Predicted={pred:.2f}, Actual={y_test[i]:.2f} -> {action}")

score = model.score(X_test, y_test)
print(f"\nR² Score: {score:.4f}")
print(f"Trading Signal (from correlation): {trading_signal}")
print(f"\nThis model looks great! R² is high!")
print(f"But it's CHEATING — it uses future data it won't have in production.")
