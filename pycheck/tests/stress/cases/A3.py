# id: A3
# kind: leak
# expect: T012
# accept: 
# allow: 
# xfail: 
# desc: shuffled KFold on price data

import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv('prices.csv', parse_dates=['date'])
cv = KFold(n_splits=5, shuffle=True)
