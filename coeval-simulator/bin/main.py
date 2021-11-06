"""
Author: Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca)

Generates a coeval boxes using 21cmFAST and simulates wedge removal
Data set is saved as a single h5 file
"""

import os
import sys
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
    

def make_dir(dirname: str) -> None:
    """Wrapper for os.mkdir"""
    try:
        os.mkdir(dirname)
    except FileExistsError:
        print(f"\n----\tWarning: {dirname} is not empty\t----\n")


def make_coeval_dset(all_params: dict, 
                     args: argparse.ArgumentParser) -> None:
    """
    Generates a dataset of coeval boxes and associated wedge-filtered boxes,
    saving it to an h5 file.
    ----------
    Params:
    :all_params: (dict) All the parameters specified in config file
    """


    with h5py.File(f"scratch/datasets/{args.name}.h5", "w") as hf:

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
            redshifts = np.linspace(7., 8.5, 9)

        original_boxes = CM.generate_coeval_boxes(redshifts)
        wedge_filtered_boxes = FM.remove_wedge(original_boxes, redshifts)

        hf.create_dataset(f"original_boxes", data=original_boxes)
        hf.create_dataset(f"wedge_filtered_boxes", data=wedge_filtered_boxes)
        hf.create_dataset(f"redshifts", data=redshifts)

        # Store p21c initial conditions to dataset
        hf.attrs["p21c_initial_conditions"] = str(CM.ic_kwargs)

        # On success
        LOGGER.info("\n----------\n")
        LOGGER.info(f"h5py file created at scratch/datasets/{args.name}.h5")
        LOGGER.info("Contents:")
        for k in hf.keys():
            LOGGER.info("\t'{}', shape: {}".format(k, hf[k].shape))
        LOGGER.info("p21cmFAST params: {}".format(pprint.pformat(CM.ic_kwargs)))
        LOGGER.info("\n----------\n")


def parse_args():
    """Handle the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="name of dataset. The dataset will be saved to scratch/datasets/<name>.h5")
    parser.add_argument("config_file", help="filepath to .yml configuration file")
    parser.add_argument("--make_coeval_dset", action="store_true",\
                        help="generate coeval boxes")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    
    # Handle the command line arguments
    args = parse_args()
    all_params = read_params_from_yml_file(args.config_file)

    if args.make_coeval_dset:
        make_coeval_dset(all_params, args)
