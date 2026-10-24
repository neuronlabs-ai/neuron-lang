# id: B6
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: manual full-sample z-score before split

df['z'] = (df['x'] - df['x'].mean()) / df['x'].std()
X_train, X_test = X[:800], X[800:]
