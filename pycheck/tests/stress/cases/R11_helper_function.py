# id: R11_helper_function
# kind: leak
# expect: T010
# accept: 
# allow: 
# xfail: 
# desc: z-score hidden in a helper function

def normalize(df):
    df['ret_z'] = (df['ret'] - df['ret'].mean()) / df['ret'].std()
    return df

df = normalize(df)
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
