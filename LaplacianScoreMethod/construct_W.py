import numpy as np


def construct_W(X, **kwargs):
    """
    Construct the affinity matrix W through different ways

    Notes:
    IF kwargs is null, use the default parameter settings;
    IF kwargs is not null, construct the affinity matrix according to parameters in kwargs.

    Input:
    X: {numpy array}, shape (n_samples, n_features) input data

    kwargs: {dictionary}
        parameters to construct different affinity matrix W:
        y: {numpy array}, shape (n_samples, 1)
            the true label information needed under the 'supervised' neighbor mode
        metric: {string}
            choices for different distance measures
            'euclidean' - use euclidean distance
            'cosine' - use cosine distance (default)
        neighbor_mode: {string}
            indicates how to construct the graph
            'knn' - put an edge between two nodes if and only if they are among the k nearest neighbors
                    of each other (default);
            'supervised' - put an edge between two nodes if they belong to same class and they are among
                    the k nearest neighbors of each other.
        weight_mode: {string}
            indicates how to assign weights for each edge in the graph
            'binary' - 0-1 weighting, every edge receives weight of 1 (default)
            'heat_kernel' - if nodes i and j are connected, put weight W_ij = ....
            'cosine' - if nodes i and j are connected, put weight cosine (x_i, x_j).
                        this weight mode can only be used under 'cosine' metric.
        k: {int}
            choices for the number of neighbors


    Output:
    W: {sparse matrix}, output affinity matrix W
    """

    # default metric is 'cosine'
    if 'metric' not in kwargs.keys():
        kwargs['metric'] = 'cosine'

    # default neighbor mode is 'knn' and default neighbor size is 5
    if 'neighbor_mode' not in kwargs.keys():
        kwargs['neighbor_mode'] = 'knn'
    if kwargs['neighbor_mode'] == 'knn' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'y' not in kwargs.keys():
        print('Warning: label is required in the supervised neighbor Mode!!!')
        exit(0)

    # default weight mode is 'binary', default t in heat kernel mode is 1
    if 'weight_mode' not in kwargs.keys():
