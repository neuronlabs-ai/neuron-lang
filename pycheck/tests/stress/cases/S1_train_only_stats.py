# id: S1_train_only_stats
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: stats from train only, applied to both

cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
mu, sd = train['ret'].mean(), train['ret'].std()
train['ret_z'] = (train['ret'] - mu) / sd
test['ret_z'] = (test['ret'] - mu) / sd
