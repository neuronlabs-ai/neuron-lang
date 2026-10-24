# id: N0_readme_headline
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: README headline example, verbatim

df['target'] = df['close'].shift(-1)
