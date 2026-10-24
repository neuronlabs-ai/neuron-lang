# id: B16
# kind: leak
# expect: F002
# accept: F001
# allow: 
# xfail: GAP: partition provenance is name-based; test rows stored in a variable called X_train are not recognised
# desc: misleading names: test data named X_train, fit on it

X_train = df[df['date'] > cutoff]
scaler.fit_transform(X_train)
