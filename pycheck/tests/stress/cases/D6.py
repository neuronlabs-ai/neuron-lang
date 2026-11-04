# id: D6
# kind: leak
# expect: T012
# accept: T020 
# allow: 
# xfail: 
# desc: shuffled KFold, temporal only via sort_values('timestamp')

from sklearn.model_selection import KFold

df = df.sort_values('timestamp')
cv = KFold(n_splits=5, shuffle=True)
