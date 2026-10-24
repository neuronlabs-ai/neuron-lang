# id: R5_minmax
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail:
# desc: full-sample statistic before split

df['x'] = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
