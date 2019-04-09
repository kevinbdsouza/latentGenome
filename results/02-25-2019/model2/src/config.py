import os


class Config:
    def __init__(self):
        self.network = 'lstm'
        self.load_weights = False

        self.input_size_encoder = 8
        self.hidden_size_encoder = 3
        self.cell_assay_embed_size = self.hidden_size_encoder

        self.input_size_decoder = self.hidden_size_encoder
        self.hidden_size_decoder = self.hidden_size_encoder
        self.output_size_decoder = self.input_size_encoder

        self.learning_rate = 10e-3

        self.cut_seq_len = 100
        self.base_pair_resolution = 25
        self.use_dna_seq = False

        self.fasta_path = "/opt/data/latent/data/dna"
        self.epigenome_npz_path_train = '/opt/data/latent/data/npz/subset_test_5_celltypes/2_cells/train'
        self.epigenome_npz_path_test = '/opt/data/latent/data/npz/subset_test_5_celltypes/2_cells/test'
        self.epigenome_bigwig_path = '/opt/data/latent/data/bigwig'

        self.model_dir = '/home/kevindsouza/Documents/projects/latent/src/saved_model'
        self.config_base = 'config.yaml'
        self.tensorboard_log_base = 't_log'
        self.config_file = os.path.join(self.model_dir, self.config_base)
        self.tensorboard_log_path = os.path.join(self.model_dir, self.tensorboard_log_base)

        if not os.path.exists(self.tensorboard_log_path):
            os.makedirs(self.tensorboard_log_path)

        self.data_dir = '.data/'
        self.num_epochs = 10

        self.chr21_len = 1924774
