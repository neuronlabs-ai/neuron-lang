"""
PyCheck VS Code Live Interactive Test
-------------------------------------
Save this file (Ctrl + S) or hover over any red/yellow squiggly line!
Open the Problems panel (Ctrl + Shift + M) to see all diagnostics.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

df = pd.DataFrame({
    'close': [100.0, 102.0, 101.5, 103.0],
    'sector': ['Tech', 'Tech', 'Energy', 'Energy'],
    'target': [1, 0, 1, 0]
})

# 1. Temporal Lookahead Leak (T001) -> Red squiggly line
df['future_target'] = df['close'].shift(1)

# 2. Preprocessing Data Leakage (T010) -> Red squiggly line
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['close']])

# 3. Time-Series Shuffling Leak (T012) -> Red squiggly line
cv = KFold(n_splits=5, shuffle=True)

# 4. Brand-New In-Sample Target Encoding Leak (T018) -> Red squiggly line
df['sector_mean'] = df.groupby('sector')['target'].transform('mean')

# 5. Causal Confusion (C001) -> Yellow squiggly line
correlation = df[['close', 'target']].corr()
print("Execution finished! (Notice how runtime Python ran without error, but PyCheck caught all silent ML flaws above)")
