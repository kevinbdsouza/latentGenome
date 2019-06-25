import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import re

'''
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
'''

'''
# read wiggle file 
rex = re.compile('fixedStep chrom=(\S+) start=(\d+) step=(\d+)')

dir_path = "/data2/latent/data/interpretation/phylogenetic_scores/"
chr_21_length = 46, 709, 983
p_score = np.zeros((48119900, 1))

file_path = dir_path + "chr21.phyloP100way.wigFix"

start = 0
count = 0

f = open(file_path)
for line in f:
    if line.startswith('fixedStep'):
        m = rex.match(line.strip())
        if m is None:
            raise (Exception, "Not a valid fixedStep line:\n{0}\nAbort!!".format(line))

        chr = m.group(1)
        start = int(m.group(2))
        step = int(m.group(3))

        count = 0
    else:
        score = float(line.strip())
        p_score[start + count] = score
        count += 1

np.save(dir_path + "p_score.npy", p_score)

'''

dir_path = "/data2/latent/data/interpretation/phylogenetic_scores/"
npy_path = dir_path + "p_score.npy"

p_score = np.load(npy_path)
window = 25
reduced_p_score = np.zeros((int(len(p_score) / window), 1))
count = 0

for cut in range(int(len(p_score)/window)):
    reduced_p_score[count] = p_score[cut * window: (cut + 1) * window].mean()
    count += 1

print("done")
