# id: F6_label_hstack
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: np.hstack([X, label]) -> fit

import numpy as np

label = df['close'].shift(-1)
X2 = np.hstack([X, label.values.reshape(-1, 1)])
model.fit(X2, y)
