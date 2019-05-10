import pandas as pd
import re


class Rep_timing:
    def __init__(self):
        self.rep_data = None
        self.huvec = "RT_HUVEC_Umbilical"
        self.imr90 = "RT_IMR90_Lung"
        self.k562 = "RT_K562_Bone"
        self.nhek = "RT_NHEK_Keratinocytes_Int92817591_hg38"
        self.gm12878 = "RT_GM12878_Lymphocyte_Int90901931_hg38"

    def get_rep_data(self, fire_path):


        fires = pd.read_pickle(self.fire_pkl)

        ''' decimate by 25 to get the positions at 25 bp resolution '''
        fires["start"] = fires["start"] // 25
        fires["end"] = fires["end"] // 25

        fire_chosen = fires.iloc[:, 0:10]

        self.fire_data = fire_chosen

    def filter_rep_data(self, chrom_rep):
        fire_data_chr = self.fire_data.loc[self.fire_data['chr'] == chrom_rep].reset_index(drop=True)

        fire_data_chr['GM12878_l'] = 0
        fire_data_chr['H1_l'] = 0
        fire_data_chr['IMR90_l'] = 0
        fire_data_chr['MES_l'] = 0
        fire_data_chr['MSC_l'] = 0
        fire_data_chr['NPC_l'] = 0
        fire_data_chr['TRO_l'] = 0

        fire_data_chr.loc[fire_data_chr['GM12878'] >= 0.5, 'GM12878_l'] = 1
        fire_data_chr.loc[fire_data_chr['H1'] >= 0.5, 'H1_l'] = 1
        fire_data_chr.loc[fire_data_chr['IMR90'] >= 0.5, 'IMR90_l'] = 1
        fire_data_chr.loc[fire_data_chr['MES'] >= 0.5, 'MES_l'] = 1
        fire_data_chr.loc[fire_data_chr['MSC'] >= 0.5, 'MSC_l'] = 1
        fire_data_chr.loc[fire_data_chr['NPC'] >= 0.5, 'NPC_l'] = 1
        fire_data_chr.loc[fire_data_chr['TRO'] >= 0.5, 'TRO_l'] = 1

        fire_labeled = fire_data_chr[
            ['chr', 'start', 'end', 'GM12878_l', 'H1_l', 'IMR90_l', 'MES_l', 'MSC_l', 'NPC_l', 'TRO_l']]

        return fire_labeled

    def augment_tad_negatives(self, cfg, tad_df):

        neg_df = pd.DataFrame(columns=['start', 'end', 'target'])

        for i in range(tad_df.shape[0]):
            diff = tad_df.iloc[i]['end'] - tad_df.iloc[i]['start']

            start_neg = tad_df.iloc[i]['start'] - diff
            end_neg = tad_df.iloc[i]['start'] - 1

            if i == 0 or start_neg > tad_df.iloc[i - 1]['end']:
                neg_df = neg_df.append({'start': start_neg, 'end': end_neg, 'target': 0},
                                       ignore_index=True)

        tad_updated = pd.concat([tad_df, neg_df]).reset_index(drop=True)

        return tad_updated


if __name__ == '__main__':
    rep_timing_path = "/data2/latent/data/downstream/replication_timing"

    chrom_rep = 21
    cell_names = ['GM12878', 'HUVEC', 'IMR90', 'K562', 'NHEK']

    rep_ob = Rep_timing()
    rep_ob.get_rep_data(rep_timing_path)
    fire_labeled = rep_ob.filter_rep_data(chrom_rep)

    print("done")
