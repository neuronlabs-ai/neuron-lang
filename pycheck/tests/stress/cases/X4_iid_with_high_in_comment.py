# id: X4_iid_with_high_in_comment
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail:
# desc: iid script with the word 'high' in a comment

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# accuracy is high on this dataset
cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(clf, iris_X, iris_y, cv=cv)
