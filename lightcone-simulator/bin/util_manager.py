import re
import sys
import h5py
import logging

import os
import typing
from typing import Optional, List
import numpy as np

def init_logger(f: str, 
                name: str) -> logging.Logger:
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")
    file_handler = logging.FileHandler(f)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = init_logger("test.log", __name__)

class UtilManager:

    def __init__(self):
        self.dset_attrs = {}
        self.random_seeds = {}

    """Manager class for miscellaneous I/O operations"""
    def load_data_from_h5(self,
                          filename: str,
                          augment: bool = False,
                          cube_shape: tuple = (128, 128, 128),
                          start: int = 0,
                          end: Optional[int]= None) -> List[np.ndarray]: 
        """
        Loads data from h5 file. Assumes that each dataset in the h5 file has
        the shape (2, num_samples, *cube_shape). The first element on the 0
        axis are the training samples and the second are their true values.
        -----
        Params:
        :filename: name of h5 data file. Assumes that each dataset in the h5 file
        stores its data as [X, Y], with shape (2, num_samples, *cube_shape).
        :augment: augments the dataset if True
        :cube_shape: shape of a single training sample
        :start: used if you only want a subset of the data
        :end: used if you only want a subset of the data
        -----
        Loads lightcones from h5py file. Assumes h5py file has the datasets:
            'lightcones' --> original lightcones (mean of each slice removed) 
            'wedge_filtered_lightcones' --> lighcones minus wedge
            'redshifts' --> redshifts values for los axis.

        Returns: X and Y data, redshift values for each pixel along the los.
        """

        with h5py.File(filename, "r") as hf:

            # Check we have the required datasets
            datasets = list(hf.keys())
            logger.info(f"Datasets: {datasets}")
            assert "wedge_filtered_lightcones" in datasets and \
                   "lightcones" in datasets and \
                   "binarized_lightcones" in datasets and \
                   "redshifts" in datasets and \
                   "random_seeds" in datasets, \
                   "Failed to extract datasets from h5py file."

            _X = np.array(hf["wedge_filtered_lightcones"][:], dtype=np.float32)
            _Y = np.array(hf["lightcones"][:], dtype=np.float32)
            _B = np.array(hf["binarized_lightcones"][:], dtype=np.float32)
            _Z = np.array(hf["redshifts"][:], dtype=np.float32)
            _rs = np.array(hf["random_seeds"][:], dtype=np.float32)

            for k,v in hf.attrs.items():
                self.dset_attrs[k] = v

        # Assert no nan values
        assert np.isnan(np.sum(_X)) == False
        assert np.isnan(np.sum(_Y)) == False
        assert np.isnan(np.sum(_Z)) == False
        assert np.isnan(np.sum(_B)) == False

        assert _X.shape[-3:] == cube_shape, f"expected {cube_shape}, got {_X.shape[-3:]}"
        assert _Y.shape == _X.shape
        assert _X.shape[1] == _Z.shape[0], "Must be one redshift per los-pixel"
        assert _X.shape[0] == _rs.shape[0], "Must be one random seed per lightcone"
        
        X = np.reshape(_X, (-1, *cube_shape))
        Y = np.reshape(_Y, (-1, *cube_shape))
        B = np.reshape(_B, (-1, *cube_shape))
        Z = np.reshape(_Z, (-1))
        self.random_seeds = np.reshape(_rs, (-1))

        return X[start:end], Y[start:end], B[start:end], Z[start:end]


    def write_str(self, 
                  s: str, 
                  filename: str) -> None:
        """Writes string to file"""
        assert type(s) is str

        with open(filename,"w") as f:
            f.write(s)
            f.close()

