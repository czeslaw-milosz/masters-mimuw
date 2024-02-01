import datetime
import logging
import os

from config import config
from utils import data_utils

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    logging.info(f"{datetime.datetime.now()}: Starting count matrices calculation...")
    matrices = data_utils.extract_count_matrices(
        config.SIGPROFILER_INPUT_DIR,
        reference_genome=config.REFERENCE_GENOME,
        project_name=config.SIGPROFILER_PROJECT_NAME 
    )

    logging.info(f"{datetime.datetime.now()}: Saving calculated matrices...")
    os.makedirs(os.path.join(config.SIGPROFILER_INPUT_DIR, "matrices"), exist_ok=True)
    for matrix_name, m in matrices.items():
        m.to_csv(
            os.path.join(config.SIGPROFILER_INPUT_DIR, "matrices", f"sigmatrix_{matrix_name}.csv"),
            index=True,
            header=True
        )
