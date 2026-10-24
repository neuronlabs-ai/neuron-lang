# id: X2_iid_with_flower_column
# kind: clean
# expect: NONE
# accept: 
# allow: 
# xfail:
# desc: iid script with a 'flower' column ('low' substring)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df['flower'] = df['species']
cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(clf, iris_X, iris_y, cv=cv)
