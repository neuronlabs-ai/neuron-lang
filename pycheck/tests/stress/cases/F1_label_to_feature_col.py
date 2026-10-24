# id: F1_label_to_feature_col
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: df['feat'] = label -> fit

label = df['close'].shift(-1)
df['feat'] = label
model.fit(df[['feat']], y)
