# id: B12
# kind: leak
# expect: T005
# accept: 
# allow: 
# xfail: 
# desc: pct_change(periods=-1) keyword

df['r'] = df['close'].pct_change(periods=-1)
