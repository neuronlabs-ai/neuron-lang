# id: S7_eda_print_stats
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: printing full-sample stats for EDA

print(df['x'].mean(), df['x'].std())
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
