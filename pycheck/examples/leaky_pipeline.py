"""
leaky_pipeline.py — The Most Common Data Science Mistake

This is a pattern seen in thousands of Kaggle notebooks, trading blogs,
and university ML courses. It looks completely reasonable. Every step
makes intuitive sense. And it will silently give you fake performance
numbers every single time.

Run:   pycheck leaky_pipeline.py
Fix:   compare with clean_pipeline.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── 1. Load time-series data ─────────────────────────────────────────────────
np.random.seed(42)
n = 2000
dates = pd.date_range("2020-01-01", periods=n, freq="D")

df = pd.DataFrame({
    "date":   dates,
    "open":   np.random.normal(100, 5, n),
    "high":   np.random.normal(105, 5, n),
    "low":    np.random.normal(95, 5, n),
    "close":  np.cumsum(np.random.normal(0, 1, n)) + 100,
    "volume": np.random.exponential(1_000_000, n),
}).set_index("date")

# ── 2. Feature Engineering ────────────────────────────────────────────────────

# BUG 1 (T001): .shift(-1) accesses the NEXT day's close price.
# The model learns to predict tomorrow using... tomorrow's price.
# Backtest accuracy: 97%. Live accuracy: 51%.
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# BUG 2 (T003): Rolling mean computed on the FULL dataset.
# Future observations contribute to the rolling mean used as a feature.
df["ma_20"]  = df["close"].rolling(20).mean()
df["ma_50"]  = df["close"].rolling(50).mean()

# BUG 3 (T014): .diff(-5) computes price change FORWARD 5 days.
df["momentum"] = df["close"].diff(-5)

# BUG 4 (T007): .bfill() fills missing values with FUTURE values.
df["ma_20"] = df["ma_20"].bfill()
df.dropna(inplace=True)

# ── 3. Scale features — BEFORE splitting ─────────────────────────────────────

X = df[["open", "high", "low", "close", "volume", "ma_20", "ma_50", "momentum"]]
y = df["target"]

# BUG 5 (T010): fit_transform() on the FULL dataset.
# The scaler computes mean/std using test data.
# Test distribution is now embedded in the training features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Split and cross-validate ───────────────────────────────────────────────

# BUG 6 (T002): train_test_split() shuffles time-series data.
# Future examples randomly appear in the training set.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# BUG 7 (T012): KFold shuffles across time — each fold leaks future into past.
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ── 5. Train & Evaluate ───────────────────────────────────────────────────────

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# BUG 8 (U005): .score() on training data — this is memorization, not accuracy.
train_score = model.score(X_train, y_train)
test_score  = model.score(X_test,  y_test)
preds = model.predict(X_test)

print(f"Train accuracy: {train_score:.1%}")   # Will be near 100%
print(f"Test  accuracy: {test_score:.1%}")    # Will be artificially inflated

# ── What PyCheck reports: ─────────────────────────────────────────────────────
# ERROR[T001]: .shift(-1) accesses data 1 rows INTO THE FUTURE
# ERROR[T003]: .rolling() computed on full dataset may include future data
# ERROR[T014]: .diff(-5) computes difference using future data
# ERROR[T007]: .bfill() fills missing values with FUTURE data
# ERROR[T010]: .fit_transform() on full dataset leaks test statistics into training
# ERROR[T002]: train_test_split() shuffles time-series data, leaking future into training
# ERROR[T012]: KFold() shuffles time-series data across folds, leaking future into training
# ERROR[U005]: .score(X_train) evaluates on training data — this is overfitting, not validation
#
# Summary: 8 error(s)
# These 8 error(s) would be COMPILE-TIME ERRORS in NEURON
# Python detected: 0 of these issues at runtime
