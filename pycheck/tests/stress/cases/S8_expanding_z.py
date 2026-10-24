# id: S8_expanding_z
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: expanding z-score shifted by one

df['z'] = (df['x'] - df['x'].expanding().mean().shift(1)) / df['x'].expanding().std().shift(1)
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
