# id: B5
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: reverse-slice reverse cumulative (future sum)

df['fut_sum'] = df['ret'][::-1].cumsum()[::-1]
