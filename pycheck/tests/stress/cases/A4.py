# id: A4
# kind: leak
# expect: T007
# accept: 
# allow: 
# xfail: 
# desc: bfill

df['p'] = df['price'].bfill()
