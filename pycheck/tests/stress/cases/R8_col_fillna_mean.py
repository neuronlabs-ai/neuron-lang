# id: R8_col_fillna_mean
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: full-sample statistic before split

df['x'] = df['x'].fillna(df['x'].mean())
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
