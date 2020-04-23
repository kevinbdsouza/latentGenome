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

    def plot_all(self, path):
        avocado_rna, avocado_pe, avocado_fire, avocado_rep, lstm_rna, lstm_pe, lstm_fire, lstm_rep = self.get_dict(path)

        # self.plot_rna_seq(path, lstm_rna, avocado_rna)
        self.plot_pe(path, avocado_pe, lstm_pe)
        # self.plot_fire(path, avocado_fire, lstm_fire)
        # self.plot_rep(path, avocado_rep, lstm_rep)

    def get_dict(self, path):

        avocado_rna = np.load(path + "avocado_rna.npy").item()
        lstm_rna = np.load(path + "lstm_rna.npy").item()

        avocado_pe = np.load(path + "avocado_pe.npy").item()
        lstm_pe = np.load(path + "lstm_pe.npy").item()

        avocado_fire = np.load(path + "avocado_fire.npy").item()
        lstm_fire = np.load(path + "lstm_fire.npy").item()

        avocado_rep = np.load(path + "avocado_rep_timing.npy").item()
        lstm_rep = np.load(path + "lstm_rep.npy").item()

        return avocado_rna, avocado_pe, avocado_fire, avocado_rep, lstm_rna, lstm_pe, lstm_fire, lstm_rep

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

    def plot_rna_seq(self, path, lstm_rna, avocado_rna):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_rna)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_rna)

        # value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)

        plt.figure(figsize=(14, 6))
        plt.ylim(0, 1)
        plt.xticks(rotation=90, fontsize=14)
        # plt.title('Gene Expression', fontsize=12)
        # plt.xlabel('Cell Types')
        plt.ylabel('MAP', fontsize=14)
        plt.yticks(fontsize=14)

        label_list = ['avocado', 'lstm']
        color_list = ['blue', 'red']

        values = [value_list_avocado, value_list_lstm]

        for i, label in enumerate(label_list):
            plt.scatter(key_list_avocado, values[i], label=label, c=color_list[i])

        plt.legend(fontsize=15)
        plt.show()
        print("done")
        # plt.savefig(path + 'lstm_rna.png')

    def plot_pe(self, path, avocado_pe, lstm_pe):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_pe)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_pe)

        value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)

        df = pd.DataFrame(
            zip(key_list_avocado * 4, ["avocado"] * 4 + ["lstm"] * 4, value_list_avocado + value_list_lstm),
            columns=["cell types", "labels", "MAP"])
        palette = {"avocado": "C0", "lstm": "C4"}
        plt.figure()
        sns.set(font_scale=1.3)
        sns.barplot(x="cell types", hue="labels", y="MAP", palette=palette, data=df)

        plt.legend(fontsize=15)
        plt.show()
        print("done")
        # plt.savefig(path + 'map_pe.png')

    def plot_fire(self, path, avocado_fire, lstm_fire):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_fire)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_fire)

        value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)

        df = pd.DataFrame(
            zip(key_list_avocado * 7, ["avocado"] * 7 + ["lstm"] * 7, value_list_avocado + value_list_lstm),
            columns=["cell types", "labels", "MAP"])
        palette = {"avocado": "C0", "lstm": "C4"}
        plt.figure()
        sns.set(font_scale=1.3)
        sns.barplot(x="cell types", hue="labels", y="MAP", palette=palette, data=df)

        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(path + 'map_fire.png')

    def plot_tad(self, path, tad_dict):
        key_list, value_list = self.get_lists(tad_dict)

        plt.figure()
        plt.bar(range(len(key_list)), value_list, align='center')
        plt.xticks(range(len(key_list)), key_list)
        plt.title('Topologically Associated Domains (TADs)')
        # plt.xlabel('Cell Types')
        plt.ylabel('MAP')
        plt.legend()
        plt.savefig(path + 'tad.png')

    def plot_rep(self, path, avocado_rep, lstm_rep):
        key_list_avocado, value_list_avocado = self.get_lists(avocado_rep)
        key_list_lstm, value_list_lstm = self.get_lists(lstm_rep)

        value_list_lstm = self.reorder_lists(key_list_lstm, key_list_avocado, value_list_lstm)

        df = pd.DataFrame(
            zip(key_list_avocado * 5, ["avocado"] * 5 + ["lstm"] * 5, value_list_avocado + value_list_lstm),
            columns=["cell types", "labels", "MAP"])
        palette = {"avocado": "C0", "lstm": "C4"}
        plt.figure()
        plt.ylim(0.85, 1)
        sns.set(font_scale=1.3)
        sns.barplot(x="cell types", hue="labels", y="MAP", palette=palette, data=df)

        plt.legend(fontsize=15)
        plt.show()
        # plt.savefig(path + 'map_rep.png')

    def plot_hidden(self, path, hidden_list):
        map_hidden = np.load(
            "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden/map_hidden.npy")
        map_hidden[7] = 0.913
        map_2_layer = [0.71, 0.76, 0.825, 0.857, 0.87, 0.885, 0.912, 0.915]
        map_dropout = [0.65, 0.71, 0.785, 0.812, 0.851, 0.862, 0.89, 0.90]
        map_no_ln = [0.657, 0.715, 0.795, 0.82, 0.86, 0.869, 0.892, 0.905]
        map_bidir = [0.692, 0.750, 0.82, 0.847, 0.862, 0.882, 0.91, 0.914]

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
        plt.savefig(path + 'hidden.png')
        plt.show()

        pass


if __name__ == "__main__":
    setup_logging()
    config_base = 'config.yaml'
    result_base = 'down_images'
    model_path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21"

    cfg = get_config(model_path, config_base, result_base)
    plot_ob = PlotMap(cfg)

    plot_ob.plot_all(plot_ob.path)

    hidden_list = [6, 12, 24, 36, 48, 60, 96, 110]
    # plot_ob.plot_hidden(plot_ob.path, hidden_list)

    print("done")
