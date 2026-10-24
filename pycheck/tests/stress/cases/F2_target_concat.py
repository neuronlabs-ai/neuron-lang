# id: F2_target_concat
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: pd.concat([X, target]) -> fit

import pandas as pd

target = df['close'].shift(-1)
X = pd.concat([X, target], axis=1)
model.fit(X, y)
