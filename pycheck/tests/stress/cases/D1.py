# id: D1
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: label-named future value copied into a feature

label = df['close'].shift(-1)
df['feat'] = label
model.fit(df[['feat']], y)
