# id: F4_label_rolling_feature
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: rolling feature built from label

label = df['close'].shift(-1)
feats = label.rolling(5).mean()
model.fit(feats.to_frame(), y)
