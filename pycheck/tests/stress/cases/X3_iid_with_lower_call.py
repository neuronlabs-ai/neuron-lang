# id: X3_iid_with_lower_call
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail:
# desc: iid script calling .lower()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

name = label.lower()
cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(clf, iris_X, iris_y, cv=cv)
