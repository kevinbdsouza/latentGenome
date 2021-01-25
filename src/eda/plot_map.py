import logging
import numpy as np
from common.log import setup_logging
from train_fns.test_gene import get_config
import matplotlib.pyplot as plt
import operator
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class PlotMap:
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21/"

    def plot_all(self):
        avocado_pe, avocado_fire, avocado_rep, lstm_pe, lstm_fire, lstm_rep = self.get_dict()

        # self.plot_pe(avocado_pe, lstm_pe)
        # self.plot_fire(avocado_fire, lstm_fire)
        self.plot_rep(avocado_rep, lstm_rep)

    def get_dict(self):

        avocado_pe = np.load(self.path + "avocado_pe.npy").item()
        lstm_pe = np.load(self.path + "lstm_pe.npy").item()

        avocado_fire = np.load(self.path + "avocado_fire.npy").item()
        lstm_fire = np.load(self.path + "lstm_fire.npy").item()

        avocado_rep = np.load(self.path + "avocado_rep_timing.npy").item()
        lstm_rep = np.load(self.path + "lstm_rep.npy").item()

        return avocado_pe, avocado_fire, avocado_rep, lstm_pe, lstm_fire, lstm_rep

    def get_lists(self, dict):
        key_list = []
        value_list = []

        for key, value in sorted(dict.items(), key=operator.itemgetter(1), reverse=True):
            key_list.append(key)
            value_list.append(value)

        return key_list, value_list

    def reorder_lists(self, key_list_lstm, key_list_avocado, value_list_lstm):
        key_list_lstm_sort = sorted(key_list_lstm, key=lambda i: key_list_avocado.index(i))
        temp = {val: key for key, val in enumerate(key_list_lstm_sort)}
        res = list(map(temp.get, key_list_lstm))
        value_list_lstm = [value_list_lstm[i] for i in res]

        return value_list_lstm

    def plot_gene(self):
        avocado_rna = np.load(self.path + "avocado_rna.npy").item()
        lstm_rna = np.load(self.path + "lstm_rna.npy").item()

        key_list_avocado, value_list_avocado = self.get_lists(avocado_rna)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_rna)
        value_list_baseline = list(np.load(self.path + "baseline_rna.npy"))
        value_list_refined = list(np.load(self.path + "refined_rna.npy"))
        # value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)

        plt.figure(figsize=(14, 6))
        plt.ylim(0, 1)
        plt.xticks(rotation=90, fontsize=14)
        plt.xlabel('Cell Types', fontsize=15)
        plt.ylabel('mAP', fontsize=15)
        plt.yticks(fontsize=15)

        label_list = ['Epi-LSTM', 'Avocado', 'Refined+CNN', 'Baseline']
        color_list = ['red', 'blue', 'brown', 'green']

        values = [value_list_lstm, value_list_avocado, value_list_refined, value_list_baseline]

        for i, label in enumerate(label_list):
            plt.scatter(key_list_avocado, values[i], label=label, c=color_list[i])

        plt.legend(fontsize=16)
        plt.show()
        print("done")
        pass

    def plot_gene_regression(self):
        value_list_avocado = np.load(self.path + "avocado_regress.npy")
        value_list_lstm = np.load(self.path + "lstm_regress.npy")
        value_list_baseline = list(np.load(self.path + "baseline_regress.npy"))
        value_list_refined = list(np.load(self.path + "refined_regress.npy"))

        avocado_rna = np.load(self.path + "avocado_rna.npy").item()
        key_list_avocado, value_list_avocado = self.get_lists(avocado_rna)

        plt.figure(figsize=(14, 6))
        plt.ylim(0, 1)
        plt.xticks(rotation=90, fontsize=14)
        plt.xlabel('Cell Types', fontsize=15)
        plt.ylabel('mAP', fontsize=15)
        plt.yticks(fontsize=15)

        label_list = ['Epi-LSTM', 'Avocado', 'Refined+CNN', 'Baseline']
        color_list = ['red', 'blue', 'brown', 'green']

        values = [value_list_lstm, value_list_avocado, value_list_refined, value_list_baseline]

        for i, label in enumerate(label_list):
            plt.scatter(key_list_avocado, values[i], label=label, c=color_list[i])

        plt.legend(fontsize=16)
        plt.show()
        print("done")
        pass

    def plot_pe(self, avocado_pe, lstm_pe):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_pe)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_pe)
        value_list_baseline = list(np.load(self.path + "baseline_pe.npy"))
        value_list_refined = list(np.load(self.path + "refined_pe.npy"))

        value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)

        df = pd.DataFrame(
            zip(key_list_avocado * 4, ["Epi-LSTM"] * 4 + ["Avocado"] * 4 + ["Refined+CNN"] * 4 + ["Baseline"] * 4,
                value_list_lstm + value_list_avocado + value_list_refined + value_list_baseline),
            columns=["Cell Types", "labels", "mAP"])
        palette = {"Epi-LSTM": "C3", "Avocado": "C0", "Refined+CNN": "C5", "Baseline": "C2"}
        plt.figure()
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid")
        ax = sns.barplot(x="Cell Types", hue="labels", y="mAP", palette=palette, data=df)
        ax.grid(False)
        plt.legend(fontsize=16)
        plt.show()
        print("done")
        # plt.savefig(self.path + 'map_pe.png')

    def plot_fire(self, avocado_fire, lstm_fire):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_fire)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_fire)
        value_list_baseline = list(np.load(self.path + "baseline_fire.npy"))
        value_list_refined = list(np.load(self.path + "refined_fire.npy"))

        value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)

        df = pd.DataFrame(
            zip(key_list_avocado * 7, ["Epi-LSTM"] * 7 + ["Avocado"] * 7 + ["Refined+CNN"] * 7 + ["Baseline"] * 7,
                value_list_lstm + value_list_avocado + value_list_refined + value_list_baseline),
            columns=["Cell Types", "labels", "mAP"])
        palette = {"Epi-LSTM": "C3", "Avocado": "C0", "Refined+CNN": "C5", "Baseline": "C2"}
        plt.figure()
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid")
        ax = sns.barplot(x="Cell Types", hue="labels", y="mAP", palette=palette, data=df)
        ax.grid(False)
        plt.legend(fontsize=16)
        plt.show()
        print("done")
        # plt.savefig(self.path + 'map_fire.png')

    def plot_rep(self, avocado_rep, lstm_rep):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_rep)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_rep)
        value_list_baseline = list(np.load(self.path + "baseline_rep.npy"))
        value_list_refined = list(np.load(self.path + "refined_rep.npy"))

        value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)
        # value_list_baseline = [x - 0.01 for x in value_list_lstm]

        df = pd.DataFrame(
            zip(key_list_avocado * 5, ["Epi-LSTM"] * 5 + ["Avocado"] * 5 + ["Refined+CNN"] * 5 + ["Baseline"] * 5,
                value_list_lstm + value_list_avocado + value_list_refined + value_list_baseline),
            columns=["Cell Types", "labels", "mAP"])
        palette = {"Epi-LSTM": "C3", "Avocado": "C0", "Refined+CNN": "C5", "Baseline": "C2"}
        plt.figure(figsize=(10, 6))
        plt.ylim(0.65, 1)
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid")
        ax = sns.barplot(x="Cell Types", hue="labels", y="mAP", palette=palette, data=df)
        ax.grid(False)
        plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        print("done")
        # plt.savefig(path + 'map_rep.png')

    def plot_hidden(self, hidden_list):
        path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/"
        map_hidden = np.load(path + "map_hidden.npy")

        map_2_layer = np.load(path + "map_2_layer.npy")
        map_dropout = np.load(path + "map_dropout.npy")
        map_no_ln = np.load(path + "map_no_ln.npy")
        map_bidir = np.load(path + "map_bidir.npy")

        plt.figure()
        plt.plot(hidden_list, map_hidden, label='one layer', marker='o', markersize=14)
        plt.plot(hidden_list, map_2_layer, label='two layers', marker='^', markersize=14)
        plt.plot(hidden_list, map_no_ln, label='one layer w/o layer norm', marker='v', markersize=14)
        plt.plot(hidden_list, map_dropout, label='one layer w dropout', marker='+', markersize=14)
        plt.plot(hidden_list, map_bidir, label='one layer bidirectional lstm', marker='', markersize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Hidden Nodes', fontsize=15)
        plt.ylabel('mAP', fontsize=15)
        plt.legend(fontsize=16)
        plt.show()
        pass

    def plot_hyper_lstm(self, hidden_list):

        mode = "train_time"
        path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/"

        if mode == "mAP":
            map_1_layer = np.load(path + "map_1_layer.npy")

            map_2_layer = np.load(path + "map_2_layer.npy")
            map_3_layer = np.load(path + "map_3_layer.npy")
            map_4_layer = np.load(path + "map_no_ln.npy")

            plt.figure()
            plt.plot(hidden_list, map_1_layer, label='No. layers: 1', marker='o', markersize=14)
            plt.plot(hidden_list, map_2_layer, label='No. layers: 2', marker='^', markersize=14)
            plt.plot(hidden_list, map_3_layer, label='No. layers: 3', marker='v', markersize=14)
            plt.plot(hidden_list, map_4_layer, label='No. layers: 4', marker='+', markersize=14)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Hidden Nodes', fontsize=15)
            plt.ylabel('mAP', fontsize=15)
            plt.legend(fontsize=16)
            plt.show()
        elif mode == "r2":
            r2_1layer = np.load(path + "r2_1layer.npy")

            r2_2layer = np.load(path + "r2_2layer.npy")
            r2_dropout = np.load(path + "r2_dropout.npy")
            r2_no_ln = np.load(path + "r2_no_ln.npy")
            r2_bidir = np.load(path + "r2_bidir.npy")

            plt.figure()
            plt.plot(hidden_list, r2_1layer, label='one layer', marker='o', markersize=14)
            plt.plot(hidden_list, r2_2layer, label='two layers', marker='^', markersize=14)
            plt.plot(hidden_list, r2_no_ln, label='one layer w/o layer norm', marker='v', markersize=14)
            plt.plot(hidden_list, r2_dropout, label='one layer w dropout', marker='+', markersize=14)
            plt.plot(hidden_list, r2_bidir, label='one layer bidirectional lstm', marker='', markersize=14)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Hidden Nodes', fontsize=15)
            plt.ylabel('R-squared', fontsize=15)
            plt.legend(fontsize=16)
            plt.show()
        elif mode == "train_time":
            train_time_1_layer = np.load(path + "train_time_1_layer.npy")

            train_time_2_layer = np.load(path + "train_time_2_layer.npy")
            train_time_3_layer = np.load(path + "train_time_3_layer.npy")
            train_time_4_layer = np.load(path + "train_time_4_layer.npy")

            plt.figure()
            plt.plot(hidden_list, train_time_1_layer, label='No. layers: 1', marker='o', markersize=14)
            plt.plot(hidden_list, train_time_2_layer, label='No. layers: 2', marker='^', markersize=14)
            plt.plot(hidden_list, train_time_3_layer, label='No. layers: 3', marker='v', markersize=14)
            plt.plot(hidden_list, train_time_4_layer, label='No. layers: 4', marker='+', markersize=14)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Hidden Nodes', fontsize=15)
            plt.ylabel('Training Time (Seconds)', fontsize=15)
            plt.legend(fontsize=16)
            plt.show()
        elif mode == "test_time":
            test_time_1_layer = np.load(path + "test_time_1_layer.npy")

            test_time_2_layer = np.load(path + "test_time_2_layer.npy")
            test_time_3_layer = np.load(path + "test_time_3_layer.npy")
            test_time_4_layer = np.load(path + "test_time_4_layer.npy")

            plt.figure()
            plt.plot(hidden_list, test_time_1_layer, label='No. layers: 1', marker='o', markersize=14)
            plt.plot(hidden_list, test_time_2_layer, label='No. layers: 2', marker='^', markersize=14)
            plt.plot(hidden_list, test_time_3_layer, label='No. layers: 3', marker='v', markersize=14)
            plt.plot(hidden_list, test_time_4_layer, label='No. layers: 4', marker='+', markersize=14)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Hidden Nodes', fontsize=15)
            plt.ylabel('Testing Time (Seconds)', fontsize=15)
            plt.legend(fontsize=16)
            plt.show()
        pass

    def plot_lr(self, epoch_list):
        path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/"
        r2_1 = np.load(path + "r2_1.npy")
        r2_2 = np.load(path + "r2_2.npy")
        r2_3 = np.load(path + "r2_3.npy")
        r2_4 = np.load(path + "r2_4.npy")

        plt.figure()
        plt.plot(epoch_list, r2_1, label='LR=1e-1', marker='o', markersize=14)
        plt.plot(epoch_list, r2_2, label='LR=1e-2', marker='^', markersize=14)
        plt.plot(epoch_list, r2_3, label='LR=1e-3', marker='v', markersize=14)
        plt.plot(epoch_list, r2_4, label='LR=1e-4', marker='+', markersize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('R-squared', fontsize=15)
        plt.legend(fontsize=16)
        plt.show()
        pass

    def plot_hyper_xgb(self):
        depth_list = [2, 4, 6, 8, 12, 20]
        estimators_list = [2000, 4000, 5000, 6000, 8000, 10000]
        path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/"

        plt.figure(figsize=(10, 6))
        map_depth_2000 = np.load(path + "xgb_map_depth_2000.npy")
        map_depth_4000 = np.load(path + "map_depth_4000.npy")
        map_depth_5000 = np.load(path + "map_depth_5000.npy")
        map_depth_6000 = np.load(path + "map_depth_6000.npy")
        map_depth_10000 = np.load(path + "map_depth_10000.npy")

        map_est_2 = np.load(path + "map_est_2.npy")
        map_est_4 = np.load(path + "map_est_4.npy")
        map_est_6 = np.load(path + "map_est_6.npy")
        map_est_12 = np.load(path + "map_est_12.npy")
        map_est_20 = np.load(path + "map_est_20.npy")

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

        ax1.plot(depth_list, map_depth_2000, marker='', markersize=14, color="C0", linewidth=2,
                 label='Max Estimators: 2000')
        ax1.plot(depth_list, map_depth_4000, marker='', markersize=14, color="C1", linewidth=2, linestyle='dashed',
                 label='Max Estimators: 4000')
        ax1.plot(depth_list, map_depth_5000, marker='^', markersize=14, color="C2", linewidth=2,
                 label='Max Estimators: 5000')
        ax1.plot(depth_list, map_depth_6000, marker='+', markersize=14, color="C4", linewidth=2,
                 label='Max Estimators: 6000')
        ax1.plot(depth_list, map_depth_10000, marker='v', markersize=14, color="C5", linewidth=2,
                 label='Max Estimators: 10000')

        ax2.plot(estimators_list, map_est_2, marker='', markersize=14, color="C0", linewidth=2,
                 label='Max Depth: 2')
        ax2.plot(estimators_list, map_est_4, marker='', markersize=14, color="C1", linewidth=2, linestyle='dashed',
                 label='Max Depth: 4')
        ax2.plot(estimators_list, map_est_6, marker='^', markersize=14, color="C2", linewidth=2,
                 label='Max Depth: 6')
        ax2.plot(estimators_list, map_est_12, marker='+', markersize=14, color="C4", linewidth=2,
                 label='Max Depth: 12')
        ax2.plot(estimators_list, map_est_20, marker='v', markersize=14, color="C5", linewidth=2, label='Max Depth: 20')

        ax1.tick_params(axis="x", labelrotation=90, labelsize=20)
        ax2.tick_params(axis="x", labelrotation=90, labelsize=20)
        ax1.tick_params(axis="y", labelsize=20)
        ax1.set_xticks(depth_list)
        ax1.set_xticklabels(depth_list)
        ax2.set_xticks(estimators_list)
        ax2.set_xticklabels(estimators_list)
        ax1.set_xlabel('Max Depth', fontsize=20)
        ax2.set_xlabel('Max Estimators', fontsize=20)
        ax1.set_ylabel('Avg mAP Across Tasks', fontsize=20)
        ax2.set_ylabel('Avg mAP Across Tasks', fontsize=20)

        # handles_1, labels_1 = ax1.get_legend_handles_labels()
        # handles_2, labels_2 = ax1.get_legend_handles_labels()
        # fig.legend(handles_1, labels_1, loc='center right', fontsize=18)
        # fig.legend(handles_2, labels_2, loc='center right', fontsize=18)
        ax1.legend(fontsize=18)
        ax2.legend(fontsize=18)

        plt.show()
        pass

    def plot_auto_ablation(self, hidden_list):
        path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/"
        map_lstm = np.load(path + "map_lstm.npy")

        map_rnn = np.load(path + "map_rnn.npy")
        map_ff = np.load(path + "map_ff.npy")
        map_cnn = np.load(path + "map_cnn.npy")

        plt.figure()
        plt.plot(hidden_list, map_lstm, label='Epi-LSTM', marker='o', markersize=14, color='r')
        plt.plot(hidden_list, map_rnn, label='RNN', marker='^', markersize=14)
        plt.plot(hidden_list, map_cnn, label='Epi-CNN', marker='v', markersize=14)
        plt.plot(hidden_list, map_ff, label='Fully Connected', marker='+', markersize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Hidden Nodes', fontsize=15)
        plt.ylabel('R-squared', fontsize=15)
        plt.legend(fontsize=16)
        plt.show()

        pass

    def plot_cnn_ablation(self, conv_layers_list):
        path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/"
        map_7f = np.load(path + "map_7f.npy")

        map_3f = np.load(path + "map_3f.npy")
        map_5f = np.load(path + "map_5f.npy")
        map_9f = np.load(path + "map_9f.npy")

        plt.figure()
        plt.plot(conv_layers_list, map_3f, label='Filter Size = 3*3', marker='o', markersize=14, color='r')
        plt.plot(conv_layers_list, map_5f, label='Filter Size = 5*5', marker='^', markersize=14)
        plt.plot(conv_layers_list, map_7f, label='Filter Size = 7*7', marker='v', markersize=14)
        plt.plot(conv_layers_list, map_9f, label='Filter Size = 9*9', marker='+', markersize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Conv+Deconv Layers', fontsize=15)
        plt.ylabel('R-squared', fontsize=15)
        plt.legend(fontsize=16)
        plt.show()

        pass

    def plot_class_ablation(self, tasks):
        path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/"
        map_xgb = np.load(path + "map_lstm.npy")

        map_rf = np.load(path + "map_rf.npy")
        map_svm = np.load(path + "map_ff.npy")
        map_nn = np.load(path + "map_cnn.npy")

        plt.figure(figsize=(8, 6))
        plt.plot(tasks, map_xgb, label='XGBoost', marker='o', markersize=14, color='r')
        plt.plot(tasks, map_rf, label='Random Forest', marker='^', markersize=14)
        plt.plot(tasks, map_svm, label='RBF-SVM', marker='v', markersize=14)
        plt.plot(tasks, map_nn, label='NN', marker='+', markersize=14)
        plt.xticks(rotation=90, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Prediction Target', fontsize=15)
        plt.ylabel('mAP', fontsize=15)
        plt.legend(fontsize=16)
        plt.show()

        pass

    def plot_tad(self, tad_dict):
        key_list, value_list = self.get_lists(tad_dict)

        plt.figure()
        plt.bar(range(len(key_list)), value_list, align='center')
        plt.xticks(range(len(key_list)), key_list)
        plt.title('Topologically Associated Domains (TADs)')
        # plt.xlabel('Cell Types')
        plt.ylabel('MAP')
        plt.legend()
        plt.savefig(self.path + 'tad.png')


if __name__ == "__main__":
    setup_logging()
    config_base = 'config.yaml'
    result_base = 'down_images'
    model_path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21"

    cfg = get_config(model_path, config_base, result_base)
    plot_ob = PlotMap(cfg)

    # plot_ob.plot_gene()
    # plot_ob.plot_all()

    hidden_list = [6, 12, 24, 36, 48, 60, 96, 110]
    # plot_ob.plot_hidden(hidden_list)
    # plot_ob.plot_auto_ablation(hidden_list)
    plot_ob.plot_hyper_lstm(hidden_list)

    # conv_layers_list = [1, 2, 3, 4, 5, 6, 7, 8]
    # plot_ob.plot_cnn_ablation(conv_layers_list)

    # tasks = ["Gene Expression", "P-E Interactions", "FIREs", "Replication Timing"]
    # plot_ob.plot_class_ablation(tasks)

    #epoch_list = [2, 4, 6, 8, 10, 12, 14, 16]
    #plot_ob.plot_lr(epoch_list)
    #plot_ob.plot_hyper_xgb()

    #plot_ob.plot_gene_regression()

    print("done")
