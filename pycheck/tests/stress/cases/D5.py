# id: D5
# kind: leak
# expect: T012
# accept: 
# allow: 
# xfail: 
# desc: shuffled KFold in a script whose only temporal markers are rolling/pct_change

from sklearn.model_selection import KFold

df['r'] = df['close'].pct_change()
df['vol'] = df['r'].rolling(20).std()
cv = KFold(n_splits=5, shuffle=True)
