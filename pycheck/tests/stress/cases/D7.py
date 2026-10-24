# id: D7
# kind: leak
# expect: C001
# accept: 
# allow: 
# xfail: 
# desc: correlation drives a trade decision (not a print)

if df.corr().loc['sentiment', 'ret'] > 0.3:
    buy()
