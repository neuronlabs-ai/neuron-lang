# id: N1_col_label
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: df['label'] = shift(-1) is flagged

df['label'] = df['close'].shift(-1)
