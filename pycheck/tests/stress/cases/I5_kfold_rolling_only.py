# id: I5_kfold_rolling_only
# kind: leak
# expect: T012
# accept: 
# allow: 
# xfail: 
# desc: only temporal clue is rolling/pct_change

from sklearn.model_selection import KFold

df['r'] = df['close'].pct_change()
df['vol'] = df['r'].rolling(20).std()
cv = KFold(n_splits=5, shuffle=True)
