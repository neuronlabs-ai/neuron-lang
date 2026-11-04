# id: I6_kfold_timestamp_only
# kind: leak
# expect: T012
# accept: T020 
# allow: 
# xfail: 
# desc: only temporal clue is sort_values('timestamp')

from sklearn.model_selection import KFold

df = df.sort_values('timestamp')
cv = KFold(n_splits=5, shuffle=True)
