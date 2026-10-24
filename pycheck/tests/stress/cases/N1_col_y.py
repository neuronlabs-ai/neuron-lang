# id: N1_col_y
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: df['y'] = shift(-1) is flagged

df['y'] = df['close'].shift(-1)
