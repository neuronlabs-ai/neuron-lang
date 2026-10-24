# id: N1_col_target
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: df['target'] = shift(-1) is flagged

df['target'] = df['close'].shift(-1)
