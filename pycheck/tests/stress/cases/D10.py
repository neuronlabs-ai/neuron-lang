# id: D10
# kind: leak
# expect: T004
# accept: 
# allow: 
# xfail: 
# desc: causal shift applied to a different series must not suppress

m = df['x'].expanding().mean()
lag = df['y'].shift(1)
df['z'] = m
