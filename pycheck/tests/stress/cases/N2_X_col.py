# id: N2_X_col
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: X['f'] = shift(-1)

X['f'] = df['close'].shift(-1)
