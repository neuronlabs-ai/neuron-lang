# id: D11
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: expanding mean shifted forward by name-hidden variable

m = df['x'].expanding().mean()
df['z'] = m.shift(-3)
