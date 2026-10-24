# id: A2
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: fit_transform on full dataset

X_scaled = scaler.fit_transform(X)
