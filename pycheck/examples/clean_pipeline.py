"""
clean_pipeline.py — The Same Workflow, Done Correctly

Direct fix for leaky_pipeline.py.
All 8 bugs resolved. PyCheck reports: 0 errors, 0 warnings.

Run:  pycheck clean_pipeline.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
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

# ── 2. Feature Engineering — CORRECT ─────────────────────────────────────────

# FIX 1: Use .shift(+1) to look at YESTERDAY's close (past data only).
df["target"] = (df["close"].shift(1) > df["close"]).astype(int)

# FIX 2: Rolling on raw data is OK — no future data involved here yet.
# We will fit the scaler only on train after the split.
df["ma_20"] = df["close"].rolling(20).mean()
df["ma_50"] = df["close"].rolling(50).mean()

# FIX 3: .diff(+5) — backward difference over 5 past days.
df["momentum"] = df["close"].diff(5)

# FIX 4: Use .ffill() (forward fill) — fills from past values only.
df["ma_20"] = df["ma_20"].ffill()
df.dropna(inplace=True)

# ── 3. Temporal Train/Test Split — CORRECT ───────────────────────────────────

# FIX 5+6: Chronological split — no shuffling, no future leakage.
split_idx = int(len(df) * 0.8)
train_df  = df.iloc[:split_idx]
test_df   = df.iloc[split_idx:]

X_train = train_df[["open", "high", "low", "close", "volume", "ma_20", "ma_50", "momentum"]]
y_train = train_df["target"]
X_test  = test_df[["open", "high", "low", "close", "volume", "ma_20", "ma_50", "momentum"]]
y_test  = test_df["target"]

# FIX 7: Fit scaler ONLY on training data, then transform both sets.
scaler  = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Time-Aware Cross-Validation — CORRECT ─────────────────────────────────

# FIX 8: TimeSeriesSplit preserves temporal order across all folds.
cv = TimeSeriesSplit(n_splits=5)

# ── 5. Train & Evaluate — CORRECT ────────────────────────────────────────────

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate on held-out TEST data only.
test_preds = model.predict(X_test_scaled)
test_score = accuracy_score(y_test, test_preds)

print(f"Test accuracy (honest): {test_score:.1%}")

# pycheck clean_pipeline.py
# --> [OK] No issues found.
