# id: C10
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: label construction (future target) with dropna, features lagged

label = df['close'].shift(-1)
feat = df['close'].shift(1)
