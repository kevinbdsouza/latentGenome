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

        self.plot_pe(avocado_pe, lstm_pe)
        # self.plot_fire(avocado_fire, lstm_fire)
        # self.plot_rep(avocado_rep, lstm_rep)

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
        palette = {"Epi-LSTM": "C3", "Avocado": "C0", "Refined+CNN": "C1", "Baseline": "C2"}
        plt.figure()
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid")
        ax = sns.barplot(x="Cell Types", hue="labels", y="mAP", palette=palette, data=df)
        ax.grid(False)
        plt.legend(fontsize=16)
        plt.show()
        print("done")
        # plt.savefig(path + 'map_pe.png')

    def plot_fire(self, avocado_fire, lstm_fire):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_fire)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_fire)
        value_list_baseline = list(np.load(self.path + "baseline_fire.npy"))

        value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)

        df = pd.DataFrame(
            zip(key_list_avocado * 7, ["avocado"] * 7 + ["lstm"] * 7 + ["baseline"] * 7,
                value_list_avocado + value_list_lstm + value_list_baseline),
            columns=["cell types", "labels", "MAP"])
        palette = {"avocado": "C0", "lstm": "C3", "baseline": "C2"}
        plt.figure()
        sns.set(font_scale=1.2)
        sns.barplot(x="cell types", hue="labels", y="MAP", palette=palette, data=df)

        plt.legend(fontsize=15)
        plt.show()
        print("done")
        # plt.savefig(path + 'map_fire.png')

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

    def plot_rep(self, avocado_rep, lstm_rep):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_rep)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_rep)
        value_list_baseline = list(np.load(self.path + "baseline_rep.npy"))

        value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)
        # value_list_baseline = [x - 0.01 for x in value_list_lstm]

        df = pd.DataFrame(
            zip(key_list_avocado * 5, ["avocado"] * 5 + ["lstm"] * 5 + ["baseline"] * 5,
                value_list_avocado + value_list_lstm + value_list_baseline),
            columns=["cell types", "labels", "MAP"])
        palette = {"avocado": "C0", "lstm": "C3", "baseline": "C2"}
        plt.figure()
        plt.ylim(0.65, 1)
        sns.set(font_scale=1.2)
        sns.barplot(x="cell types", hue="labels", y="MAP", palette=palette, data=df)

        plt.legend(fontsize=15, loc="center right")
        plt.show()
        print("done")
        # plt.savefig(path + 'map_rep.png')

    def plot_hidden(self, hidden_list):
        map_hidden = np.load(
            "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/map_hidden.npy")

        map_2_layer = None
        map_no_ln = None
        map_dropout = None
        map_bidir = None

        plt.figure()
        plt.plot(hidden_list, map_hidden, label='one layer')
        plt.plot(hidden_list, map_2_layer, label='two layers')
        plt.plot(hidden_list, map_no_ln, label='one layer w/o layer norm')
        plt.plot(hidden_list, map_dropout, label='one layer w dropout')
        plt.plot(hidden_list, map_bidir, label='one layer bidirectional lstm')

        # plt.xticks(range(len(key_list)), key_list)
        plt.title('MAP vs hidden nodes')
        plt.xlabel('hidden nodes')
        plt.ylabel('MAP')
        # plt.savefig(path + 'hidden.png')
        plt.legend()
        plt.savefig(self.path + 'hidden.png')
        plt.show()

        pass


if __name__ == "__main__":
    setup_logging()
    config_base = 'config.yaml'
    result_base = 'down_images'
    model_path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21"

    cfg = get_config(model_path, config_base, result_base)
    plot_ob = PlotMap(cfg)

    # plot_ob.plot_gene()
    plot_ob.plot_all()

    # hidden_list = [6, 12, 24, 36, 48, 60, 96, 110]
    # plot_ob.plot_hidden(hidden_list)

    print("done")
