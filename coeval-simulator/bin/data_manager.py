"""
Script for saving/loading h5 datafiles
@author: j-c-carr
"""

import h5py
import numpy as np


class DataManager:

    """
    Loads data from h5py file. Datasets from the h5py file are stored in the
    DataManager.data dictionary. Data from h5py Groups are store in the
    DataManager.metadata dictionary. 
    ----------
    Attributes
    :filepath:   (str) Name of h5py file.
    :data:       (dict) All h5py Datasets (retrieved as numpy arrays) loaded 
                        from the h5py file. 
    :dset_attrs: (dict) Stores h5py datafile attributes.
    :metadata:   (dict) Stores h5py Group data (retrieved as numpy arrays).
    """

    def __init__(self, filepath: str):
        assert filepath[-3:] == ".h5", "filepath must point to an h5 file."

        self.filepath = filepath
        self.data = {}
        self.dset_attrs = {}
        self.metadata = {}

        self.load_data_from_h5()
        
    def load_data_from_h5(self):
        """Loads all data from h5 file into numpy arrays"""

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
                    assert np.isnan(np.sum(v)) is False, \
                           f"Error, {k} has nan values."
                    self.data[k] = v

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
