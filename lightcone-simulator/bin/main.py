"""
Author: Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca)

Generates a data set using 21cmFAST and wedge removal from fourier.py
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

from lightcone_plot_manager import LightconePlotManager
from lightcone_manager import LightconeManager
from fourier_manager import FourierManager
from util_manager import UtilManager

# Prints all logging info to std.err
logging.getLogger().addHandler(logging.StreamHandler())



def save_dataset_attrs(hf, 
                       params: [dict, None] = None) -> None:
    """
    Params:
    :hf: (h5py.File) h5py object, dataset file in write mode
    :params: dictionary of parameters to save with data
    """
    # Save the parameters with the data
    if params is not None:
        for k,v in params.items():
            try:
                hf.attrs[str(k)] = str(v)
            except TypeError:
                continue


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


def read_params_from_yml_file(filename: str) -> dict:
    """ Reads variables from yml filename."""

    assert filename[-3:] == "yml" or filename[-4:] == "yaml"

    with open(filename) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        return params
    

def make_dir(dirname: str) -> None:
    """Wrapper for os.mkdir"""
    try:
        os.mkdir(dirname)
    except FileExistsError:
        print(f"\n----\tWarning: {dirname} is not empty\t----\n")


def plot_lightcones(all_params: dict, 
               args: argparse.ArgumentParser) -> None:
    """Generates lightcones and masks for each set of initial conditions"""

    UM = UtilManager()
    CUBE_SHAPE = (256, 128, 128)
    CUBE_DIMENSIONS = (512, 256, 256)

    # Load data
    logger.info(f"Loading data from {args.old_data}...")
    X, Y, B, redshifts = \
            UM.load_data_from_h5(f"scratch/datasets/{args.old_data}.h5",
                                 cube_shape=CUBE_SHAPE)

    mins = Y.min(axis=(2,3), keepdims=True)
    Y -= mins
    count_Y = np.count_nonzero(Y==0)
    count_B = np.count_nonzero(B==0)
    print("Ionized pixels total: ", count_B)
    print("Maybe ionized pixels in Y: ", count_Y)
    #LPM = LightconePlotManager(redshifts, CUBE_SHAPE,
    #                           CUBE_DIMENSIONS)


def make_lightcones(all_params: dict, 
               args: argparse.ArgumentParser) -> None:
    """Generates lightcones and masks for each set of initial conditions"""


    with h5py.File(f"scratch/datasets/{args.name}.h5", "w") as hf:

        LM = LightconeManager(all_params)
        FM = FourierManager()
        starting_redshift = all_params["final_starting_redshift"]
        n_los_pixels = all_params["final_lightcone_shape"][0]

        p21c_lightcones, redshifts = LM.generate_lightcones()

        # Binarize the lightcones BEFORE removing the mean along frequency axis
        start = np.where(np.floor(redshifts)==starting_redshift)[0][0]
        end = start + n_los_pixels

        binarized_lightcones = (p21c_lightcones > 0).astype(np.uint8)
        binarized_lightcones = binarized_lightcones[:, start:end]
        
        # Remove the mean along each frequency slice
        p21c_lightcones -= p21c_lightcones.mean(axis=(2, 3), keepdims=True)


        lightcones, wedge_filtered_lightcones, redshifts = \
                FM.remove_wedge_from_lightcones(p21c_lightcones, 
                                                redshifts,
                                                starting_redshift = \
                                                        starting_redshift,
                                                n_los_pixels = n_los_pixels)

        # Truncate binarized lightcones to match X shape
        hf.create_dataset("binarized_lightcones", data=binarized_lightcones)
        hf.create_dataset("lightcones", data=lightcones)
        hf.create_dataset("wedge_filtered_lightcones", 
                          data=wedge_filtered_lightcones)
        hf.create_dataset("redshifts", data=redshifts)
        hf.create_dataset("random_seeds", data=LM.params["lightcone_random_seeds"])

        # Save config data with dataset
        hf.attrs["p21c_run_lightcone_kwargs"] = \
                str(all_params["p21c_run_lightcone_kwargs"])
        hf.attrs["starting_redshift"] = str(redshifts[0])
        hf.attrs["ending_redshift"] = str(redshifts[-1])

        # Success!
        logger.info("\n----------\n")
        logger.info(f"h5py file created at scratch/datasets/{args.name}.h5")
        logger.info("Contents:")
        for k in hf.keys():
            logger.info("\t'{}', shape: {}".format(k, hf[k].shape))

        logger.info("p21.run_lightcone params:{}".format(\
                pprint.pformat(all_params["p21c_run_lightcone_kwargs"])))
        logger.info("\n----------\n")
        

def parse_args():
    """Handle the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="name of dataset. The dataset will be saved to scratch/datasets/<name>.h5")
    parser.add_argument("config_file", help="filepath to .yml configuration file")
    parser.add_argument("--make_lightcones", action="store_true", help="generate lightcones")
    parser.add_argument("--plot_lightcones", action="store_true", help="plot lightcones")
    parser.add_argument("--old_data", help="name of dataset to plot. The dataset will be loaded from scratch/datasets/<old_data>.h5")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    
    # Handle the command line arguments
    args = parse_args()
    all_params = read_params_from_yml_file(args.config_file)
    logger = init_logger("test.log", __name__)

    np.random.seed(0)

    UM = UtilManager()

    if args.make_lightcones:
        make_lightcones(all_params, args)

    if args.plot_lightcones:
        plot_lightcones(all_params, args)
