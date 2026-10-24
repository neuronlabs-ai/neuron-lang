# id: C9
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: Pipeline scaler inside CV is leak-free

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

pipe = Pipeline([('s', StandardScaler()), ('m', Ridge())])
cv_scores = cross_val_score(pipe, X, y, cv=TimeSeriesSplit(5))
