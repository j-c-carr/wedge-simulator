import h5py

import typing
from typing import Optional, List
import numpy as np

###############################################################################
#                    DataManager to load data from h5 file                    #
###############################################################################

# Access data like so:
# DM.data["wedge_filtered_brightness_temp_boxes"]
# DM.data["brightness_temp_boxes"]
# DM.data["ionized_boxes"]
# DM.data["redshifts"]
# DM.data["predicted_brightness_temp_boxes"]


class DataManager():

    """
    Loads data from h5py file. Datasets from the h5py file are stored in the
    DataManager.data dictionary. Metadata is stored in DM.metadata dictionary
    """

    def __init__(self, filepath: str):
        assert filepath[-3:] == ".h5", "filepath must point to an h5 file."

        self.filepath = filepath
        self.data = {}
        self.dset_attrs = {}
        self.metadata = {}

        self.load_data_from_h5()
        

    def load_data_from_h5(self):
        """Loads all data from h5 file, returns nothing. (Typically used just
        to observe the values in a dataset)"""

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


