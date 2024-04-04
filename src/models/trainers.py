import logging
import time

import numpy as np
import pandas as pd
import polars as pl
import pyro
import torch
import tqdm
from pyro.infer import SVI, TraceMeanField_ELBO

from config import config
from models import models
from utils import training_utils


def train_prodLDA(mutations_df: pd.DataFrame|pl.DataFrame) -> tuple[models.ProdLDA, list[float]]:
    """Train a ProdLDA model on the given data.

    Args:
        counts (pd.DataFrame|pl.DataFrame): A dataframe of counts data.
        n_signatures (int): The number of signatures to train.

    Returns:
        None
    """
    mutation_counts = torch.from_numpy(mutations_df.to_numpy().transpose())  # expected shape: (n_samples, n_features), e.g. (569, 96)

    torch.manual_seed(config.RANDOM_SEED)
    pyro.set_rng_seed(config.RANDOM_SEED)

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, using CPU; this may be very slow!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mutation_counts = mutation_counts.float().to(device)

    pyro.clear_param_store()

    prodLDA = models.ProdLDA(
        vocab_size=mutation_counts.shape[1],
        num_topics=config.N_SIGNATURES_TARGET,
        hidden=config.HIDDEN_SIZE,
        dropout=config.DROPOUT,
        loss_regularizer=config.LOSS_REGULARIZER,
        reg_lambda=config.REGULARIZER_LAMBDA
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": config.LEARNING_RATE, "betas": config.BETAS})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(np.ceil(mutation_counts.shape[0] / config.BATCH_SIZE))

    bar = tqdm.trange(config.NUM_EPOCHS)
    epoch_loss = []
    for _ in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_counts = mutation_counts[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE, :]
            loss = svi.step(batch_counts)
            running_loss += loss / batch_counts.size(0)
        epoch_loss.append(running_loss)
        bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
    
    return prodLDA, epoch_loss


def trainRRTVAE(mutations_df: pd.DataFrame|pl.DataFrame) -> tuple[models.ProdLDA, list[float]]:
    """Train a RRTVAR model on the given data.

    Args:
        counts (pd.DataFrame|pl.DataFrame): A dataframe of counts data.
        n_signatures (int): The number of signatures to train.

    Returns:
        None
    """
    mutation_counts = torch.from_numpy(mutations_df.to_numpy().transpose())  # expected shape: (n_samples, n_features), e.g. (569, 96)

    torch.manual_seed(config.RANDOM_SEED)

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, using CPU; this may be very slow!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mutation_counts = mutation_counts.float().to(device)

    model = models.RRTVAE(
        vocab_size=mutation_counts.shape[1], 
        num_topics=config.N_SIGNATURES_TARGET, 
        hidden=config.HIDDEN_SIZE,
        lambda_=config.LAMBDA, 
        delta=config.DELTA, 
        prior_alpha=config.PRIOR_ALPHA, 
        device=device
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

    epoch_loss = []
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        all_indices = torch.randperm(mutation_counts.size(0)).split(config.BATCH_SIZE)
        loss_epoch = 0.0
        kld_epoch = 0.0
        model.train()  # switch to training mode
        
        for batch_indices in all_indices:
            batch_indices = batch_indices.to(device)
            inputs = mutation_counts[batch_indices]

            _, (loss, kld_loss) = model(inputs, avg_loss=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()
            kld_epoch += kld_loss.item()
        
        epoch_loss.append(loss_epoch / len(all_indices))
        print('====> Epoch: {} Average loss: {:.4f}, KL Div: {:.4f}'.format(
          epoch, loss_epoch / len(all_indices), kld_epoch / len(all_indices)))
             
        end_time = time.time()
        epoch_mins, epoch_secs = training_utils.epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    
    return model, epoch_loss

