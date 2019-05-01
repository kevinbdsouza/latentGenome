import os


class Config:
    def __init__(self):
        self.network = 'lstm'
        self.load_weights = False

        self.input_size_encoder = 1030
        self.hidden_size_encoder = 24

        self.cell_assay_embed_size = self.hidden_size_encoder

        self.input_size_decoder = self.hidden_size_encoder
        self.hidden_size_decoder = self.hidden_size_encoder
        self.output_size_decoder = self.input_size_encoder

        self.learning_rate = 1e-2

        self.cut_seq_len = 100
        self.base_pair_resolution = 25
        self.use_dna_seq = False

        self.fasta_path = "/opt/data/latent/data/dna"
        self.epigenome_npz_path_train = '/opt/data/latent/data/npz/ch_21_arc_sin_znorm'
        self.epigenome_npz_path_test = '/opt/data/latent/data/npz/ch_21_arc_sin_znorm'
        self.epigenome_bigwig_path = '/opt/data/latent/data/bigwig'

        self.model_dir = '/home/kevindsouza/Documents/projects/latentGenome/src/saved_model/model_all_ca'
        self.config_base = 'config.yaml'
        self.tensorboard_log_base = 't_log'
        self.config_file = os.path.join(self.model_dir, self.config_base)
        self.tensorboard_log_path = os.path.join(self.model_dir, self.tensorboard_log_base)

        if not os.path.exists(self.tensorboard_log_path):
            os.makedirs(self.tensorboard_log_path)

        self.data_dir = '.data/'
        self.num_epochs = 5

        self.chr21_len = 1926400
        self.downstream_df_columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13',
                                      'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24',
                                      'target', 'gene_id']
