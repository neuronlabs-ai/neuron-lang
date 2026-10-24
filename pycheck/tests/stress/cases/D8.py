# id: D8
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: future label then used to build a rolling feature

label = df['close'].shift(-1)
feats = label.rolling(5).mean()
