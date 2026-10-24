# id: B1
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: shift offset via variable

k = -1
df['y'] = df['close'].shift(k)
