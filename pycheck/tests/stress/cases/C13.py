# id: C13
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: train-named var fit_transform (correct) after chronological split

cut = int(len(df) * 0.8)
X_train, X_test = X[:cut], X[cut:]
X_train_s = scaler.fit_transform(X_train)
