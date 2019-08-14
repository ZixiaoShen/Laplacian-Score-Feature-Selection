import numpy as np



def lap_score(X, **kwargs):
    """
    This function implements the laplacian score feature selection, steps are as follows:
    1. Construct the affinity matrix W if it is not specified;
    2. For the r-th feature, we define fr = X(:,r), D = diag(W * ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr' * D * ones) * ones/(ones ' * D * ones)
    4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat)

    Input:
    -------
    X: (numpy array), shape (n_samples, n_features) input data
    kwargs: {dictionary} W: {sparse matrix}, shape (n_samples, n_samples)
            input affinity matrix

    Output:
    -------
    score: {numpy array}, shape (n_features,)  laplacian score for each feature
    """

    if 'W' not in kwargs.keys():
        W = construct_W(X)

