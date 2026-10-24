# id: F7_label_via_helper_func
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: label passed through helper into features

def add_feat(X, s):
    X['f'] = s
    return X
label = df['close'].shift(-1)
X = add_feat(X, label)
model.fit(X, y)
