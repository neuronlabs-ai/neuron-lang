# id: L3_label_used_as_fit_target
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail:
# desc: label only reaches fit() as y

label = df['close'].shift(-1)
X = df[['lag1', 'lag2']]
model.fit(X, label)
