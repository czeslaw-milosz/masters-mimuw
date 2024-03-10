import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from sigProfilerPlotting import plotSBS
from matplotlib import image as mpimg

from models.trainers import train_prodLDA
from config import config
from utils.data_utils import get_nearest_signatures


if __name__ == "__main__":
    # test CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
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

    P = model.beta()
    P_prob = (P / P.sum(dim=1).unsqueeze(-1)).numpy()
    nearest_signatures = get_nearest_signatures(P_prob, cosmic, scipy.spatial.distance.cosine)
    print(list(nearest_signatures.items()))
    discovered_signatures = pd.DataFrame(P_prob.transpose(), index=pd.Index(data=cosmic.index, name="MutationType"))

    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("ProdLDA loss history")
    plt.savefig(config.LOSS_PLOT_PATH)
    if config.SAVE_MODEL:
        torch.save(model.state_dict(), config.PRODLDA_MODEL_PATH)

    discovered_plots = plotSBS(
        discovered_signatures,
        "tmp/", "project", plot_type="96", savefig_format="pil_image", dpi=200, percentage=True
    )
    plt.rcParams["figure.dpi"] = 240
    plt.rcParams['figure.figsize'] = [6.4, 8]
    fig, axs = plt.subplots(len(discovered_plots), 2)
    fig.suptitle("ProbLDA-discovered signatures vs. cosine-nearest COSMIC signatures")
    for i in range(len(discovered_plots)):
        axs[i,0].imshow(discovered_plots[i])
        axs[i,0].axis("off")
        axs[i,0].text(0.7, 0.7, f"cos similarity: {str(round(nearest_signatures[i][1], 2))}", size=8)
        axs[i,1].imshow(mpimg.imread(f"./data/cosmic/plots/{nearest_signatures[i][0]}.png"))
        axs[i,1].axis("off")
    plt.savefig(config.SIGNATURES_PLOT_PATH)
