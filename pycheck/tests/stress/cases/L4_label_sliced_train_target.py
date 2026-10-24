# id: L4_label_sliced_train_target
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail:
# desc: label[:cut] as y, features lagged

label = df['close'].shift(-1)
cut = int(len(df) * 0.8)
model.fit(df[['lag1']][:cut], label[:cut])
