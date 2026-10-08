# Real-world financial workflow published on multiple algorithmic trading blogs
# Demonstrates subtle data leakage: scaling full dataset before train/test split

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Simulated price data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'Open': np.random.normal(150, 10, n),
    'High': np.random.normal(155, 10, n),
    'Low': np.random.normal(145, 10, n),
    'Close': np.random.normal(150, 10, n),
    'Volume': np.random.normal(5000000, 100000, n),
})

# 2. Create Target with negative shift (future lookahead)
df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
df.dropna(inplace=True)

# 3. Features and Preprocessing — THE LEAK:
# scaler.fit_transform() is called on the ENTIRE dataset before train/test split!
# This leaks future test set distribution (mean/variance) into training features.
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 5. Model training & prediction
model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
