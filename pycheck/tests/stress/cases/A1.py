# id: A1
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: literal shift(-1)

df['y'] = df['close'].shift(-1)
