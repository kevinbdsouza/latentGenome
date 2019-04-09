import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import average_precision_score
from common.log import setup_logging
from downstream.run_downstream import DownstreamTasks
from train_fns.test_gene import get_config
from downstream.rna_seq import RnaSeq

setup_logging()
config_base = 'config.yaml'
result_base = 'down_images'
model_path = "/home/kevindsouza/Documents/projects/latent/results/03-04-2019_n/downstream/model4"

file_name = "/home/kevindsouza/Documents/projects/latent/results/03-04-2019_n/downstream/feat_E123.pkl"
feature_matrix = pd.read_pickle(file_name)
feature_matrix.label = feature_matrix.label.astype(int)

average_precisions = np.zeros(5)
for i in range(5):
    msk_test = np.random.rand(len(feature_matrix)) < 0.8
    X_train_val = feature_matrix[msk_test].reset_index(drop=True)
    X_test = feature_matrix[~msk_test].reset_index(drop=True)

    msk_val = np.random.rand(len(X_train_val)) < 0.8
    X_train = X_train_val[msk_val].reset_index(drop=True)
    X_valid = X_train_val[~msk_val].reset_index(drop=True)

    model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1] - 1, 4), max_depth=6)
    model.fit(X_train.iloc[:, 0:3], X_train[:]["label"], eval_set=[(X_valid.iloc[:, 0:3], X_valid.iloc[:]["label"])],
              eval_metric='map',
              early_stopping_rounds=20,
              verbose=False)

    y_hat = model.predict_proba(X_test.iloc[:, 0:3])

    average_precisions[i] = average_precision_score(X_test.iloc[:]["label"], y_hat[:, 1])

mean_map = average_precisions.mean()
print("done")
