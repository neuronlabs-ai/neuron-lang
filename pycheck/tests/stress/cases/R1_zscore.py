# id: R1_zscore
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: full-sample statistic before split

df['ret_z'] = (df['ret'] - df['ret'].mean()) / df['ret'].std()
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
