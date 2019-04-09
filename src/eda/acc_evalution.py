import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve
import itertools
from matplotlib.patches import Rectangle
import collections
from common.helper.db_query import NasDB
import operator
from data_handling.handlers.siamese_handler.patch_dict_maker import PatchDictMaker
import pandas as pd
from pathlib import Path


def get_log(file_path):
    with open(file_path, 'r') as f:
        log = json.load(f)
    y_t = log['y_true']
    y_p = log['y_predict']
    y_t = np.array([tt for t in y_t for tt in t])
    y_p = np.array([pp for p in y_p for pp in p])
    return y_t, y_p


def get_sids(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    data = np.array(data)
    return data


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_roc(y_t, y_p):
    fpr, tpr, threshold = roc_curve(y_t, y_p)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([-0.1, 1.1], [-0.1, 1.1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()


def plot_histrogram(y_t, y_p):
    cmap = plt.get_cmap('jet')
    similar = cmap(0.5)
    dissimilar = cmap(0.8)

    plt.hist(y_p[y_t == 1], bins=100, histtype='step', range=(0, 1), color=similar)
    plt.hist(y_p[y_t == 0], bins=100, histtype='step', range=(0, 1), color=dissimilar)
    plt.yscale('log')
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in [similar, dissimilar]]
    labels = ['similar', "dissimilar"]
    plt.legend(handles, labels)
    plt.tight_layout()


def plot_precision_recall_curve(y_t, y_p):
    precision, recall, _ = precision_recall_curve(y_t, y_p)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.formataverage_precision))
    plt.tight_layout()


def confusion_index(y_t, y_p):
    tp = (y_p > 0.5) & (y_t == 1)
    tn = (y_p <= 0.5) & (y_t == 0)
    fp = (y_p > 0.5) & (y_t == 0)
    fn = (y_p <= 0.5) & (y_t == 1)
    return tp, tn, fp, fn


def get_pairs(sids, df, group_by=None):
    if group_by == 'drone':
        cols = 'drone_name'
    else:
        cols = 'group_name'
    pair = [df.loc[sids[:, i], cols].values for i in range(2)]
    pair = list(zip(*pair))
    pair = [tuple(sorted(x)) for x in pair]
    pair = [x[0] + ', ' + x[1] for x in pair]
    return pair


def get_pair_distribution(pairs):
    return dict(collections.Counter(pairs))


def plot_pair_distribution(pair_distribution):
    sorted_d = sorted(pair_distribution.items(), key=operator.itemgetter(1), reverse=True)
    labels = [k for (k, v) in sorted_d]
    values = [v for (k, v) in sorted_d]
    plt.bar(range(len(pair_distribution)), values, tick_label=labels)
    # plt.xticks(rotation=30)
    plt.tight_layout()


def plot_incorrect_patches(patches, dfs, output_folder):
    row_number = 4
    pair_number = patches.shape[1]

    iters = -(-pair_number // row_number)  # calculate the ceiling
    for i in range(iters):
        df_subset = [df.iloc[row_number * i: row_number * (i + 1), :] for df in dfs]
        df_subset = [df.reset_index(drop=True) for df in df_subset]
        patches_subset = patches[:, row_number * i: row_number * (i + 1), :, :]
        f, axarr = plt.subplots(row_number, 2, figsize=(6, 12), clear=True)
        for j in range(row_number):
            axarr[j, 0].imshow(patches_subset[0, j, :, :])
            axarr[j, 0].set_title(df_subset[0].loc[j, 'drone_name'], size=10)
            axarr[j, 1].imshow(patches_subset[1, j, :, :])
            axarr[j, 1].set_title(df_subset[1].loc[j, 'drone_name'], size=10)
        plt.tight_layout()
        filename = Path(output_folder) / 'incorrect_labelled_pair_{}'.format(i)
        plt.savefig(str(filename))
        plt.close()


if __name__ == '__main__':
    # reading data
    y_t, y_p = get_log('/mnt/nas/USD/train_fns/siamese_network_simpler_network_3_groups/testing_log.json')
    # training_sids = get_sids('/mnt/nas/USD/train_fns/siamese_network_simpler_network_3_groups/training_sids_1423.json')
    testing_sids = get_sids('/mnt/nas/USD/train_fns/siamese_network_simpler_network_3_groups/testing_sids.json')

    # create objects
    nas = NasDB()
    patch_shape = (96, 96)
    log_every = 5
    patch_maker = PatchDictMaker(save_mode=True, patch_shape=patch_shape, output_path=None)

    # getting signals
    signal_sql = """ select s.id as s_id, s.fc_Ghz, s.bw_Mhz, s.t_start_ms, s.dwell_time_ms, s.drone_name, 
                     s.modulation, s.pwr_db, s.group_name, s.capture_name, s.snr_db, s.fading_fd, s.fading_type 
                     from signals as s """
    s_df = nas.read_query_as_df(signal_sql)
    s_df = s_df.set_index('s_id', drop=False)

    # getting captures
    capture_sql = """ select c.id as c_id, c.capture_name, c.fc_Ghz as cap_fc_Ghz, c.fs_Mhz, c.bw_Mhz as useful_bw_Mhz, 
                      c.len_ms, c.parent_address, c.spec_exist, c.stft_nfft, c.stft_window, c.stft_overlap, c.source, 
                      c.use_case, c.datetime, c.is_siamese_spec_exist from captures as c """
    c_df = nas.read_query_as_df(capture_sql)
    c_df = c_df.set_index('c_id', drop=False)

    # getting pairs
    # training_pairs = get_pairs(training_sids, s_df, group_by='drone')
    # training_pair_distribution = get_pair_distribution(training_pairs)
    testing_pairs = get_pairs(testing_sids, s_df, group_by='group')
    testing_pair_distribution = get_pair_distribution(testing_pairs)
    tp, tn, fp, fn = confusion_index(y_t, y_p)
    false_pairs = get_pairs(testing_sids[fp | fn], s_df, group_by='group')
    false_pair_distribution = get_pair_distribution(false_pairs)
    false_pair_errors = {p: 100 * v / testing_pair_distribution[p] for p, v in false_pair_distribution.items()}

    # getting confusion
    cnf_matrix = confusion_matrix(y_t, y_p > 0.5)
    np.set_printoptions(precision=2)
    print(np.sum(y_t == (y_p > 0.5)) / len(y_t))
    print(y_p)

    # get a subset of s_df
    # false_pair_df = [s_df.loc[testing_sids[fp | fn][:, i], :] for i in range(2)]
    # false_pair_df = [pd.merge(c_df, df, how='inner', on='capture_name') for df in false_pair_df]
    # patch_dicts = [patch_maker.get_patch_dict(df)[0] for df in false_pair_df]
    # patch_dicts = [{k: (v - v.min()) / (v.max() - v.min()) for k, v in patch.items()} for patch in patch_dicts]
    # ids = [df.index.values.tolist() for df in false_pair_df]
    # patches = [[patch[k] for k in id] for (id, patch) in zip(ids, patch_dicts)]
    # patches = [[np.expand_dims(array, axis=0) for array in patch] for patch in patches]
    # patches = [np.concatenate(patch, axis=0) for patch in patches]
    # patches = [np.expand_dims(patch, axis=0) for patch in patches]
    # patches = np.concatenate(patches, axis=0)  # (left/right, event, h, w)

    #
    # plot_incorrect_patches(patches, false_pair_df, '/mnt/nas/USD/train_fns/siamese_network_simpler_network_3_groups')

    # Plotting metrics
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=['dissimilar', 'similar'], normalize=False)
    #
    # plt.figure()
    # plot_histrogram(y_t, y_p)
    #
    # plt.figure()
    # plot_precision_recall_curve(y_t, y_p)
    #
    # plt.figure()
    # plot_roc(y_t, y_p)

    plt.figure()
    plot_pair_distribution(false_pair_errors)

    # plt.figure()
    # plot_pair_distribution(training_pair_distribution)
    plt.show()
