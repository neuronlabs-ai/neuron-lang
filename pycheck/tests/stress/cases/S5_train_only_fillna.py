# id: S5_train_only_fillna
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: impute with train mean after split

cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
mu = train.mean()
train = train.fillna(mu)
test = test.fillna(mu)
