# id: D9
# kind: leak
# expect: T004
# accept: T001
# allow: 
# xfail: 
# desc: expanding().mean().shift(-1) (future), not a causal lag

df['m'] = df['close'].expanding().mean().shift(-1)
