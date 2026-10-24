# id: S4_cross_sectional_zscore
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail:
# desc: per-date cross-sectional z-score is causal

df['z'] = df.groupby('date')['x'].transform(lambda s: (s - s.mean()) / s.std())
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
