import scipy.io
#from skfeature.function.similarity_based import lap_score
#from skfeature.utility import construct_W
from LaplacianScoreMethod import construct_W
from LaplacianScoreMethod import lap_score

mat = scipy.io.loadmat('/home/zealshen/DATA/DATAfromASU/FaceImageData/COIL20.mat')
X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]

# construct affinity matrix
kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
W = construct_W.construct_W(X, **kwargs_W)

# obtain the scores of features
score = lap_score.lap_score(X, W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score)

print('Score:', score)
print('Index:', idx)
