import os
import numpy as np

DATA_DIR = "data"
MUTATIONS_FILE_NAME = "simple_somatic_mutation.open.BRCA-EU.tsv"
COSMIC_FILE_PATH = "data/cosmic/COSMIC_v3.4_SBS_GRCh37.txt"

SIGPROFILER_PROJECT_NAME = "BRCA-EU"
REFERENCE_GENOME = "GRCh37"
SIGPROFILER_INPUT_DIR = "data/brca"

COUNTS_DF_PATH = "data/brca/matrices/sigmatrix_96.csv"

RANDOM_SEED = 2137
N_SIGNATURES_TARGET = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3 * 2
BETAS = (0.95, 0.999)
DROPOUT = 0.3
HIDDEN_SIZE = 128
NUM_EPOCHS = 200
LOSS_REGULARIZER = "cosine"

EXPERIMENT_NAME = "prodLDApluscosine"
PRODLDA_MODEL_PATH = f"data/models/{EXPERIMENT_NAME}model.pt"
LOSS_PLOT_PATH = f"data/plots/{EXPERIMENT_NAME}_loss.png"
SIGNATURES_PLOT_PATH = f"data/plots/{EXPERIMENT_NAME}_signatures.png"
SAVE_MODEL = False
