# id: D4
# kind: leak
# expect: T002
# accept: 
# allow: 
# xfail: 
# desc: iid-sounding name wrapping daily price data

import pandas as pd
from sklearn.model_selection import train_test_split

iris = pd.read_csv('daily_prices.csv', parse_dates=['date'])
tr, te = train_test_split(iris, test_size=0.2)
