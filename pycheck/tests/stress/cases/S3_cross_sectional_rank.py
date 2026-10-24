# id: S3_cross_sectional_rank
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail:
# desc: per-date cross-sectional rank is causal

df['r'] = df.groupby('date')['x'].rank(pct=True)
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
