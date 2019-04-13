import numpy as np
import xgboost
from sklearn.metrics import average_precision_score


def celltype_baseline(y):
    y_hat = np.zeros_like(y) + y.mean()
    return average_precision_score(y, y_hat)


def make_predictions(X, y):
    np.random.seed(0)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    average_precisions = np.zeros(20)
    for i in range(5):
        X_test = X[i::20]
        X_valid = X[(i + 1) % 20::20]
        X_train = np.concatenate([X[j::20] for j in range(20) if j != i and j != (i + 1) % 20])

        y_test = y[i::20]
        y_valid = y[(i + 1) % 20::20]
        y_train = np.concatenate([y[j::20] for j in range(20) if j != i and j != (i + 1) % 20])

        model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 4), max_depth=6)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20,
                  verbose=False)

        y_hat = model.predict_proba(X_test)
        average_precisions[i] = average_precision_score(y_test, y_hat[:, 1])

    return average_precisions


n = 10

X = np.load('/home/kevindsouza/Documents/projects/latent/baseline/avocado/data/RNAseq.E017.AvocadoFac.npy')

X5 = X[:n]

y = np.load('/home/kevindsouza/Documents/projects/latent/baseline/avocado/data/RNAseq.E017.y.npy')[:n]
y = (y >= 0.5).astype(int)

map = make_predictions(X5, y)
mean_map = map.mean()
print("done")
