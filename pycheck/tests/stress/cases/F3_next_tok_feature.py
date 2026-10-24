# id: F3_next_tok_feature
# kind: leak
# expect: F001
# accept: T001
# allow: 
# xfail: 
# desc: next_tok in a trading script -> feature

next_tok = df['close'].shift(-1)
X['f'] = next_tok
model.fit(X, y)
