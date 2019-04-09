import numpy
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style('whitegrid')
import itertools
from baseline.avocado.avocado.model import Avocado

# load data into dictionary
celltypes = ['E003', 'E017', 'E065', 'E116', 'E117']
assays = ['H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K4me1']

data = {}
for celltype, assay in itertools.product(celltypes, assays):
    if celltype == 'E065' and assay == 'H3K4me3':
        continue

    filename = '/home/kevindsouza/Documents/projects/latent/baseline/avocado/data/{}.{}.pilot.arcsinh.npz'.format(
        celltype, assay)
    data[(celltype, assay)] = numpy.load(filename)['arr_0']

# create train_fns
model = Avocado(celltypes, assays, n_layers=1, n_genomic_positions=1126469, n_nodes=128, n_assay_factors=24,
                n_celltype_factors=32,
                n_25bp_factors=25, n_250bp_factors=40, n_5kbp_factors=45, batch_size=10000)

model.summary()
model.fit(data, n_epochs=5, epoch_size=113)
y_hat = model.predict("E065", "H3K4me3")  # Make new predictions

''' find mse '''
start, end = 12750, 15000
x = numpy.arange(start * 25 / 1000., end * 25 / 1000., 25 / 1000.)
y_true = numpy.load("/home/kevindsouza/Documents/projects/latent/baseline/avocado/data/E065.H3K4me3.pilot.arcsinh.npz")[
    'arr_0']
mse_global = ((y_hat - y_true) ** 2).mean()
baseline_mse = ((y_true - y_true.mean()) ** 2).mean()

''' plotting '''
plt.figure(figsize=(14, 4))
plt.subplot(211)
plt.title("How good is our E065 H3K4me3 Imputation? Global MSE: {:4.4}, Global Baseline MSE: {:4.4}".format(mse_global,
                                                                                                            baseline_mse),
          fontsize=16)
plt.fill_between(x, 0, y_true[start:end], color='b', label="Roadmap Signal")
plt.legend(fontsize=14)
plt.ylabel("Signal Value", fontsize=14)
plt.ylim(0, 7)
plt.xlim(start * 25 / 1000., end * 25 / 1000.)

plt.subplot(212)
plt.fill_between(x, 0, y_hat[start:end], color='g', label="Avocado Imputation")
plt.legend(fontsize=14)
plt.ylabel("Signal Value", fontsize=14)
plt.xlabel("Genomic Coordinate (kb)", fontsize=14)
plt.ylim(0, 7)
plt.xlim(start * 25 / 1000., end * 25 / 1000.)
plt.show()
