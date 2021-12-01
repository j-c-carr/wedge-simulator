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
    """Manager class for miscellaneous I/O operations"""
        self.data = {}
        self.dset_attrs = {}
        self.metadata = {}


    def load_data_from_h5(self, filepath):
        """Loads all data from h5 file, returns nothing. (Typically used just
        to observe the values in a dataset)"""

        self.filepath = filepath

        with h5py.File(self.filepath, "r") as hf:

            for k in hf.keys():

                # AstroParams are stored in an h5py group
                if isinstance(hf[k], h5py.Group):
                    self.metadata[k] = {}
                    for k2 in hf[k].keys():
                        v = np.array(hf[k][k2], dtype=np.float32)
                        self.metadata[k][k2] = v

                # Lightcone data is stored as h5py datasets
                if isinstance(hf[k], h5py.Dataset):
                    v = np.array(hf[k][:], dtype=np.float32)
                    assert np.isnan(np.sum(v)) == False, \
                            f"Error, {k} has nan values."
                    self.data[k] = v
            #self.data["redshifts"].reshape(-1) 

            # Load metadata from h5 file
            for k, v in hf.attrs.items():
                self.dset_attrs[k] = v

        # Print success message
        print("\n----------\n")
        print(f"data loaded from {self.filepath}")
        print("Contents:")
        for k, v in self.data.items():
            print("\t{}, shape: {}".format(k, v.shape))
        print("\nMetadata:")
        for k in self.metadata.keys():
            print(f"\t{k}")
        print("\n----------\n")
        print("\nDataset Attributes:")
        for k in self.dset_attrs.keys():
            print(f"\t{k}")
        print("\n----------\n")

    def write_str(self, 
                  s: str, 
                  filename: str) -> None:
        """Writes string to file"""
        assert type(s) is str

        with open(filename,"w") as f:
            f.write(s)
            f.close()

