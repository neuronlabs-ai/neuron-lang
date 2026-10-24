# id: C7
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: shuffled KFold on iid iris data

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=0)
iris_scores = cross_val_score(clf, iris_X, iris_y, cv=cv)
