# id: C2
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: TimeSeriesSplit

from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
