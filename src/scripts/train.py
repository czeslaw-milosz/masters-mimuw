import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.trainers import train_prodLDA
from config import config


if __name__ == "__main__":
    # Load the data
    mutations_df = pd.read_csv(
        config.COUNTS_DF_PATH,
        index_col=0
    )

    cosmic = pd.read_csv(
        config.COSMIC_FILE_PATH, 
        sep="\t"
    ).rename(
        {"Type": "MutationType"}, 
        axis="columns"
    ).set_index("MutationType")

    assert np.all(mutations_df.index == cosmic.index)
    model, loss = train_prodLDA(mutations_df)

    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("ProdLDA loss history")
    plt.savefig(config.LOSS_PLOT_PATH)
    torch.save(model.state_dict(), config.PRODLDA_MODEL_PATH)
