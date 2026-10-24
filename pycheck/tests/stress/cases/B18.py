# id: B18
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: t+1 lookup via .map()/set_index key arithmetic needs value-level reasoning
# desc: leak via lookup-merge on t+1 key

df['next_close'] = df['ts'].map(df.set_index(df['ts'] - 1)['close'])
