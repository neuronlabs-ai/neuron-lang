# id: B8
# kind: leak
# expect: T010
# accept: T010
# allow: 
# xfail: 
# desc: feature selection fit on full data before CV

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

sel = SelectKBest(k=10).fit(X, y)
X_sel = sel.transform(X)
scores = cross_val_score(model, X_sel, y, cv=TimeSeriesSplit(5))
