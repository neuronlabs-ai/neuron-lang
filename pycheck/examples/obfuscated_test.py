# Completely random, bizarre variable names that no linter could ever hardcode:
import pandas as weird_pd
from sklearn.preprocessing import StandardScaler as RandomCustomScalerAlias

banana_df = weird_pd.DataFrame({'zebra': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# T001 test: shifting by -7 on 'zebra'
banana_df['future_alien_signal'] = banana_df['zebra'].shift(-7)

# T010 test: fit_transform using alias before any split
funky_scaler_obj = RandomCustomScalerAlias()
giraffe_matrix = funky_scaler_obj.fit_transform(banana_df[['zebra']])
