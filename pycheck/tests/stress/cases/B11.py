# id: B11
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: manual lead via numpy slicing

import numpy as np

df['y'] = np.append(df['close'].values[1:], np.nan)
