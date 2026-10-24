# id: R6_max_scale
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail:
# desc: full-sample statistic before split

df['x'] = df['x'] / df['x'].max()
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
