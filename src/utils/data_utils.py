import os
from typing import Callable, Dict

import numpy as np
import pandas as pd
import polars as pl
import scipy
from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matGen

from config import config


def get_nearest_signatures(
    discovered_signatures: pd.DataFrame | np.ndarray, 
    reference_signatures: pd.DataFrame,
    distance_function: Callable = scipy.spatial.distance.cosine
    ) -> Dict[int, tuple[str, float]]:
    nearest_signatures = {}
    for i in range(discovered_signatures.shape[0]):
        argmin_idx = np.argmin(
            [distance_function(
                discovered_signatures[i],
                reference_signatures.iloc[:,j]
                ) for j in range(reference_signatures.shape[1])]
            )
        nearest_signatures[i] = (reference_signatures.iloc[:, argmin_idx].name, 
                                 1 - distance_function(
                                     discovered_signatures[i], reference_signatures.iloc[:, argmin_idx]
                                     )
                                )
    return nearest_signatures
    #     nearest_signatures[i] = reference_signatures.iloc[:, np.argmin(
    #         [distance_function(
    #             discovered_signatures[i],
    #             reference_signatures.iloc[:,j]
    #             ) for j in range(reference_signatures.shape[1])]
    #         )].name
    # return {
    #     i: reference_signatures.iloc[:, np.argmin(
    #         [distance_function(
    #             discovered_signatures[i],
    #             reference_signatures.iloc[:,j]
    #             ) for j in range(reference_signatures.shape[1])]
    #         )].name
    #     for i in range(discovered_signatures.shape[0])
    # }


def create_sbs_only_dataset(input_path: str, output_dir: str) -> None:
    output_path = os.path.join(
        output_dir,
        os.path.split(input_path)[-1].replace(".tsv", ".sbs_only.tsv")
    )
    pl.scan_csv(
        input_path,
        separator="\t", 
        infer_schema_length=10_000, 
        low_memory=True
    ).filter(
        pl.col("mutation_type") == "single base substitution"
    ).collect(
        streaming=True
    ).write_csv(
        output_path,
        separator="\t"
    )
    

def extract_count_matrices(input_dir: str, 
                           reference_genome: str = config.REFERENCE_GENOME, 
                           project_name: str = config.SIGPROFILER_PROJECT_NAME
                           ) -> Dict[str, pd.DataFrame]:
    matrices = matGen.SigProfilerMatrixGeneratorFunc(
        project=project_name,
        reference_genome=reference_genome,
        path_to_input_files=input_dir,
    )
    return matrices


def read_cosmic_signatures(input_path: str = config.COSMIC_FILE_PATH) -> pd.DataFrame:
    return pd.read_csv(
        input_path, sep="\t"
        ).rename(
            {"Type": "MutationType"}, axis="columns"
        ).set_index("MutationType")