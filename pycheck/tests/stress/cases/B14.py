# id: B14
# kind: leak
# expect: T007
# accept: 
# allow: 
# xfail: 
# desc: ffill on reversed frame = bfill

df['p'] = df.iloc[::-1]['price'].ffill().iloc[::-1]
