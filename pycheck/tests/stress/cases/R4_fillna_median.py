# id: R4_fillna_median
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: full-sample statistic before split

df = df.fillna(df.median())
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
