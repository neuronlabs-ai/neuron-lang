# id: A6
# kind: leak
# expect: T016
# accept: 
# allow: 
# xfail: 
# desc: merge_asof nearest

import pandas as pd

out = pd.merge_asof(a, b, on='ts', direction='nearest')
