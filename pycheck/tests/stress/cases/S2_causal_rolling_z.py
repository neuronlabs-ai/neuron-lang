# id: S2_causal_rolling_z
# kind: clean
# expect: NONE
# accept: 
# allow: T003
# xfail:
# desc: rolling z-score is causal

df['z'] = (df['x'] - df['x'].rolling(20).mean()) / df['x'].rolling(20).std()
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
