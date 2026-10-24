# id: D2
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: target-named future series concatenated into X

import pandas as pd

target = df['close'].shift(-1)
X = pd.concat([X, target], axis=1)
model.fit(X, y)
