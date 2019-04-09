import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

pe_int_path = "/opt/data/latent/data/downstream/PE-interactions"
nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end',
                 'window_chrom', 'window_start', 'window_end', 'window_name', 'active_promoters_in_window',
                 'interactions_in_window', 'enhancer_distance_to_promoter', 'bin', 'label']

training_df = pd.read_hdf(pe_int_path + '/training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
predictors_df = training_df.drop(nonpredictors, axis=1)
labels = training_df['label']

estimator = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2, max_features='log2',
                                       random_state=0)

estimator.fit(predictors_df, labels)
importances = pd.Series(estimator.feature_importances_, index=predictors_df.columns).sort_values(ascending=False)
print(importances.head(16))
