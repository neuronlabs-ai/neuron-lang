# id: B10
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: bound-method alias

sh = df['close'].shift
df['y'] = sh(-1)
