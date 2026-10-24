# id: I1_iid_kfold_iris
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail: 
# desc: shuffled KFold on iid data

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(clf, iris_X, iris_y, cv=cv)
