# id: C1
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: shift(1) lag feature

df['lag1'] = df['close'].shift(1)
