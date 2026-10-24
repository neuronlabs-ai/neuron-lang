# id: C12
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: expanding stats only on past via shift

df['m'] = df['close'].expanding().mean().shift(1)
