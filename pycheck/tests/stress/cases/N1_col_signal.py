# id: N1_col_signal
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: df['signal'] = shift(-1) is flagged

df['signal'] = df['close'].shift(-1)
