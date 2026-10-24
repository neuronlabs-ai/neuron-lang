# id: R7_winsorize
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail:
# desc: full-sample statistic before split

lo, hi = df['x'].quantile(0.01), df['x'].quantile(0.99)
df['x'] = df['x'].clip(lo, hi)
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
