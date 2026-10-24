# id: N1_col_fwd_ret
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: df['fwd_ret'] = shift(-1) is flagged

df['fwd_ret'] = df['close'].shift(-1)
