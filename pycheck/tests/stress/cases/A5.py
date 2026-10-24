# id: A5
# kind: leak
# expect: T009
# accept: 
# allow: 
# xfail: 
# desc: centered rolling

df['m'] = df['price'].rolling(5, center=True).mean()
