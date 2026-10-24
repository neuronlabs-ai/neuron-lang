# id: B9
# kind: leak
# expect: T010
# accept: T010
# allow: 
# xfail: 
# desc: PCA.fit on full data then split

from sklearn.decomposition import PCA

pca = PCA(n_components=5).fit(X)
X_p = pca.transform(X)
