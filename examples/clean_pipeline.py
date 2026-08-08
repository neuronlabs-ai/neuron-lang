"""
clean_pipeline.py — A correctly written ML pipeline.
neuronc pycheck should find ZERO issues here.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(42)
prices = np.cumsum(np.random.randn(1000)) + 100

df = pd.DataFrame({'close': prices})

# CORRECT: target uses PAST data (shift +1, not -1)
df['target'] = df['close'].shift(1)

# CORRECT: temporal split, not random split
df = df.dropna()
split = int(len(df) * 0.8)
train = df[:split]
test = df[split:]

X_train = train[['close']].values
y_train = train['target'].values
X_test = test[['close']].values
y_test = test['target'].values

# Train
model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"R² Score: {score:.4f}")
