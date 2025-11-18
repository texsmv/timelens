from sklearn.decomposition import PCA


# Projects multivariate time series of shape N, T, D to N, 2
# 
# First reshape the input to N, T*D, then project to N, 2
def project_mts(X, reducer=None):
    N, T, D = X.shape
    X = X.reshape(N, T*D)
    if reducer is None:
        reducer = PCA(n_components=2)
    reducer.fit(X)
    return reducer.transform(X)