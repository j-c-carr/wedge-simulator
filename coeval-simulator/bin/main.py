"""
Author: Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca)

Generates a coeval boxes using 21cmFAST and simulates wedge removal
Data set is saved as a single h5 file
"""

import h5py
import yaml
import pprint
import typing
from typing import Optional
import logging
import argparse
import numpy as np

from coeval_manager import CoevalManager
from fourier_manager import FourierManager
from data_manager import DataManager

# Prints all logging info to std.err
logging.getLogger().addHandler(logging.StreamHandler())


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

LOGGER = init_logger("test.log", __name__)


def read_params_from_yml_file(filename: str) -> dict:
    """ Reads variables from yml filename."""

    assert filename[-3:] == "yml" or filename[-4:] == "yaml"

    with open(filename) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        LOGGER.debug(params)
        return params


def save_dset_to_hf(filename: str,
                    data: dict,
                    attrs: dict = {}):

    with h5py.File(f"{args.dset_dir}/{args.dset_name}.h5", "w") as hf:

        # Save datasets
        for k, v in data.items():
            hf.create_dataset(k, data=v)

        # Save attributes
        for k, v in attrs.items():
            hf.attrs[k] = str(v)

    # On success
    LOGGER.info("\n----------\n")
    LOGGER.info(f"h5py file created at {filename}")
    LOGGER.info("Contents:")
    for k in data.keys():
        LOGGER.info("\t'{}', shape: {}".format(k, data[k].shape))
    LOGGER.info("Attributes:")
    for k in attrs.keys():
        LOGGER.info("\t'{}': {}".format(k, attrs[k]))
    LOGGER.info("\n----------\n")

    
def generate_coeval_dset(all_params: dict, 
                         args: argparse.ArgumentParser) -> None:
    """
    Generates a dataset of coeval boxes and associated wedge-filtered boxes,
    saving it to an h5 file.
    ----------
    Params:
    :all_params: (dict) All the parameters specified in config file
    """

    # Load p21c initial condition parameters, if specified
    if "ic_kwargs" in all_params:
        CM = CoevalManager(all_params["ic_kwargs"])
    else:
        CM = CoevalManager()

    FM = FourierManager()

    # Default redshifts are z=7, 8.5, 9
    if "redshifts" in all_params:
        redshifts = np.array(all_params["redshifts"])
    else:
        redshifts = np.linspace(7, 8.5, 25)

    bt_boxes, xh_boxes = CM.generate_coeval_boxes(redshifts)
    wedge_filtered_boxes = FM.remove_wedge(bt_boxes, redshifts)

    save_dset_to_hf(f"{args.dset_dir}/{args.dset_name}.h5",
                    {"redshifts": redshifts,
                     "brightness_temp_boxes": bt_boxes,
                     "wedge_filtered_brightness_temp_boxes": wedge_filtered_boxes,
                     "ionized_boxes": xh_boxes},
                    attrs = {"p21c_initial_conditions": CM.ic_kwargs})


def modify_coeval_dset(old_data_loc: str,
                       all_params: dict,
                       args: argparse.ArgumentParser) -> None:
    """
    Loads the data from :old_data_loc: into a DataManager object. The data is
    stored in numpy arrays in the DataManager.data dictionary. For example,

        DM.data["brightness_temp_boxes"]
        DM.data["wedge_filtered_brightness_temp_boxes"]
        DM.data["ionized_boxes"]
        DM.data["redshifts"]

    """

    DM = DataManager(old_data_loc)

    # ...
    # ...


if __name__=="__main__":
    
    # Handle the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dset_dir", help="parent directory of dataset")
    parser.add_argument("dset_name", help="name of dataset. The dataset will be saved to <dset_dir>/<dset_name>.h5")
    parser.add_argument("config_file", help="filepath to .yml configuration file")
    parser.add_argument("--make_coeval_dset", action="store_true",\
                        help="generate coeval boxes dataset")
    parser.add_argument("--modify_coeval_dset", action="store_true",\
                        help="modify coeval boxes dataset")
    parser.add_argument("--old_data_loc", help="filepath to old dataset (.h5 file)")
    args = parser.parse_args()

    all_params = read_params_from_yml_file(args.config_file)

    if args.make_coeval_dset:
        generate_coeval_dset(all_params, args)

    elif args.modify_coeval_dset:
        assert args.old_data_loc is not None, \
                "Could not read the old_model_loc location, required."

        modify_coeval_dset(args.old_data_loc, all_params, args)


