# id: R3_fillna_mean
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: full-sample statistic before split

df = df.fillna(df.mean())
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
