# id: B2
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: shift(periods=-1) keyword

df['y'] = df['close'].shift(periods=-1)
