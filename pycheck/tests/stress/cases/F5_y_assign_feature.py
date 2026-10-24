# id: F5_y_assign_feature
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: y reused through assign()

y = df['close'].shift(-1)
X = df[['a', 'b']].assign(extra=y)
model.fit(X, y)
