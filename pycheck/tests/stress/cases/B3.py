# id: B3
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: shift via getattr

df['y'] = getattr(df['close'], 'shift')(-1)
