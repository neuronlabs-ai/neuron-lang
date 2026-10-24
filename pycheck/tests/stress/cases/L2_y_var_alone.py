# id: L2_y_var_alone
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: standalone y = shift(-1)

y = df['close'].shift(-1)
