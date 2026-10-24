# id: I4_images_named_ticks
# kind: leak
# expect: T002
# accept: 
# allow: 
# xfail: 
# desc: df named images but is tick data

import pandas as pd
from sklearn.model_selection import train_test_split

images = pd.read_csv('stock_ticks.csv', parse_dates=['ts'])
tr, te = train_test_split(images, test_size=0.2)
