# id: S6_ffill
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: ffill is causal

df = df.ffill()
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
