# id: B4
# kind: leak
# expect: T019
# accept: 
# allow: 
# xfail: 
# desc: np.roll backwards

import numpy as np

df['y'] = np.roll(df['close'].values, -1)
