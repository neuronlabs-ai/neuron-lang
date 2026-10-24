# id: D12
# kind: leak
# expect: T010
# accept: T011
# allow: 
# xfail: 
# desc: Pipeline fit on ALL rows, then scored on a slice of those same rows

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X, y)
score = pipe.score(X[800:], y[800:])
