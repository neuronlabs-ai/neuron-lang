# id: B13
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: eval() string is opaque to static analysis (certificate must say 'unverifiable', see audit tests)
# desc: leak hidden in eval string

df['y'] = eval("df['close'].shift(-1)")
