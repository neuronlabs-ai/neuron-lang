# id: A7
# kind: leak
# expect: F001
# accept: T001,T010
# allow: 
# xfail: 
# desc: taint propagation through arithmetic

fut = df['close'].shift(-1)
sig = fut - df['close']
model.fit(X, sig)
