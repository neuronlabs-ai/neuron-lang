# id: C3
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: scaler fit on train only

scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
