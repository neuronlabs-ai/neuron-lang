# ═══════════════════════════════════════════════════════════════════════
#  Sequence Model Pipeline with Data Leakage Bugs
#  Demonstrates PyCheck catching lookahead bias in Transformer features
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def build_sequence_features(df):
    # BUG 1: Lookahead leakage - shifting negative rows exposes future tokens into input features
    df['future_signal'] = df['close'].shift(-1) - df['close']

    # BUG 2: Non-causal backward diff
    df['future_diff'] = df['close'].diff(-5)

    X = df[['close', 'future_signal', 'future_diff']].values

    # BUG 3: Data leakage - scaler fit on entire dataset before split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # BUG 4: Shuffling sequence data across folds (breaks temporal order)
    cv = KFold(n_splits=5, shuffle=True)

    return X_scaled, cv
