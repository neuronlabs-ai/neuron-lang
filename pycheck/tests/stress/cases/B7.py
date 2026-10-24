# id: B7
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: fillna with full-sample mean before split

df = df.fillna(df.mean())
train, test = df[:800], df[800:]
