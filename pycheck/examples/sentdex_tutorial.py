# Source: pythonprogramming.net/training-testing-machine-learning-tutorial/
# Author: Harrison Kinsley (Sentdex) - Machine Learning with Python Series (Part 4)
# Used by millions of beginner data scientists & finance students

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Simulated stock data representing Quandl Google data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'Adj. Open': np.random.normal(500, 20, n),
    'Adj. High': np.random.normal(510, 20, n),
    'Adj. Low': np.random.normal(490, 20, n),
    'Adj. Close': np.random.normal(500, 20, n),
    'Adj. Volume': np.random.normal(1000000, 50000, n),
})

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = 30

# 1. Target creation via negative shift
df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

# 2. Scaling the ENTIRE dataset before train/test split (Leaking future distribution)
X = preprocessing.scale(X)

# 3. train_test_split with default shuffle=True on TIME-SERIES data (Leaking future into past)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Fit model on leaked data
clf = LinearRegression()
clf.fit(X_train, y_train)

# 5. Evaluate confidence
confidence = clf.score(X_test, y_test)
print("Model accuracy / confidence score:", confidence)

predictions = clf.predict(X_test)
