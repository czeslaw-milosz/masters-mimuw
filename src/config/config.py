import os

DATA_DIR = "data"
MUTATIONS_FILE_NAME = "simple_somatic_mutation.open.BRCA-EU.tsv"
COSMIC_FILE_PATH = "data/cosmic/COSMIC_v3.4_SBS_GRCh37.txt"

SIGPROFILER_PROJECT_NAME = "BRCA-EU"
REFERENCE_GENOME = "GRCh37"
SIGPROFILER_INPUT_DIR = "data/brca"

COUNTS_DF_PATH = "data/brca/matrices/sigmatrix_96.csv"

RANDOM_SEED = 2137
N_SIGNATURES_TARGET = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT = 0.2
HIDDEN_SIZE = 100
NUM_EPOCHS = 50
PRODLDA_MODEL_PATH = "data/models/prodlda.pt"
LOSS_PLOT_PATH = "data/plots/loss.png"
