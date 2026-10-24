# id: R2_rank_pct
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: full-sample statistic before split

df['vol_rank'] = df['vol'].rank(pct=True)
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
