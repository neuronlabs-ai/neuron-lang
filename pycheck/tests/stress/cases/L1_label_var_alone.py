# id: L1_label_var_alone
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: standalone label = shift(-1)

label = df['close'].shift(-1)
