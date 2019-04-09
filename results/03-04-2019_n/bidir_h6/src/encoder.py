import logging
import torch
import train_fns.lstm as lstm
import numpy as np
from torch import nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class EncoderLSTM(nn.Module):
    def __init__(self, cfg, input_size, hidden_size, gpu_id):
        self.cfg = cfg
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.gpu_id = gpu_id

    def forward(self, inputs, hidden, state):
        output, (hidden, state) = self.lstm(inputs, (hidden, state))
        return output, hidden, state

    def initHidden(self):
        h = Variable(torch.zeros(2, 1, self.hidden_size))
        c = Variable(torch.zeros(2, 1, self.hidden_size))
        return h.cuda(self.gpu_id), c.cuda(self.gpu_id)


class Embeddings:
    def __init__(self, cfg, vocab_size):
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.cell_assay_embed_size).cuda().train()
        self.init_weights = np.random.multivariate_normal(np.zeros(self.cfg.cell_assay_embed_size), np.identity(self.cfg.cell_assay_embed_size), vocab_size)
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        self.embed.weight.data.copy_(torch.from_numpy(self.init_weights))

    def celltype_assay_embedding(self, celltype_assay_id):
        return self.embed(celltype_assay_id)
