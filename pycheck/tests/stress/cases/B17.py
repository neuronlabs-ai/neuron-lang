# id: B17
# kind: leak
# expect: C002
# accept: 
# allow: 
# xfail:
# desc: target column rides in via df.copy()

X = df.copy()
model.fit(X, df['target'])
