import pandas as pd
from sklearn.linear_model import Ridge
from helper import add_features

df = add_features(pd.read_csv('prices.csv', parse_dates=['date']))
cut = int(len(df) * 0.8)
train, test = df[:cut], df[cut:]
model = Ridge().fit(train[['ret_z', 'vol_rank']], train['target'])
print(model.score(test[['ret_z', 'vol_rank']], test['target']))
