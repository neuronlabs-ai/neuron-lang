# id: N1_col_next_close
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: df['next_close'] = shift(-1) is flagged

df['next_close'] = df['close'].shift(-1)
