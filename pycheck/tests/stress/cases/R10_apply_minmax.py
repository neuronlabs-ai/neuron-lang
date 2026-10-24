# id: R10_apply_minmax
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail:
# desc: full-sample statistic before split

df[['a', 'b']] = df[['a', 'b']].apply(lambda c: (c - c.min()) / (c.max() - c.min()))
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
