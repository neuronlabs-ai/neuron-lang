# id: N1_col_feature
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: df['feature'] = shift(-1) is flagged

df['feature'] = df['close'].shift(-1)
