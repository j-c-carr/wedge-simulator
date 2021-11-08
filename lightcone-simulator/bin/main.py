"""
Author: Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca)

Generates a data set using 21cmFAST and wedge removal from fourier.py
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

from lightcone_plot_manager import LightconePlotManager
from lightcone_manager import LightconeManager
from fourier_manager import FourierManager
from util_manager import UtilManager

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


def read_params_from_yml_file(filename: str) -> dict:
    """ Reads variables from yml filename."""

    assert filename[-3:] == "yml" or filename[-4:] == "yaml"

    with open(filename) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        return params
    

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


def make_lightcone_dset(all_params: dict, 
                        args: argparse.ArgumentParser) -> None:
    """
    Generates a dataset of lightcones and associated wedge-filtered lightcones,
    saving it to an h5 file.
    ----------
    Params:
    :all_params: (dict) All the parameters specified in config file
    """


    with h5py.File(f"{args.dset_dir}/{args.dset_name}.h5", "w") as hf:

        LM = LightconeManager(all_params)
        FM = FourierManager()


        p21c_lightcones, redshifts = LM.generate_lightcones()

        # Remove the mean along each frequency slice to simulate observations
        p21c_lightcones -= p21c_lightcones.mean(axis=(2, 3), keepdims=True)

        starting_redshift = all_params["final_starting_redshift"]
        n_los_pixels = all_params["final_lightcone_shape"][0]
        lightcones, wedge_filtered_lightcones, redshifts = \
                FM.remove_wedge_from_lightcones(p21c_lightcones, 
                                                redshifts,
                                                starting_redshift = \
                                                        starting_redshift,
                                                n_los_pixels = n_los_pixels)

        hf.create_dataset("redshifts", data=redshifts)
        hf.create_dataset("lightcones", data=lightcones)
        hf.create_dataset("wedge_filtered_lightcones", 
                          data=wedge_filtered_lightcones)
        hf.create_dataset("random_seeds", 
                          data=LM.params["lightcone_random_seeds"])

        # Save config data with dataset
        hf.attrs["p21c_run_lightcone_kwargs"] = \
                str(all_params["p21c_run_lightcone_kwargs"])
        hf.attrs["starting_redshift"] = str(redshifts[0])
        hf.attrs["ending_redshift"] = str(redshifts[-1])

        # Success!
        logger.info("\n----------\n")
        logger.info(f"h5py file created at {args.dset_dir}/{args.dset_name}.h5")
        logger.info("Contents:")
        for k in hf.keys():
            logger.info("\t'{}', shape: {}".format(k, hf[k].shape))

        logger.info("p21.run_lightcone params:{}".format(\
                pprint.pformat(all_params["p21c_run_lightcone_kwargs"])))
        logger.info("\n----------\n")
        

def parse_args():
    """Handle the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("dset_dir", help="parent directory of dataset")
    parser.add_argument("dset_name", help="name of dataset. The dataset will be saved to <dset_dir>/<dset_name>.h5")
    parser.add_argument("config_file", help="filepath to .yml configuration file")
    parser.add_argument("--make_lightcone_dset", action="store_true", help="generate lightcones")
    parser.add_argument("--plot_lightcones", action="store_true", help="plot lightcones")
    parser.add_argument("--old_data", help="name of an existing dataset (only used for --plot_lightcones")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    
    # Handle the command line arguments
    args = parse_args()
    all_params = read_params_from_yml_file(args.config_file)
    logger = init_logger("test.log", __name__)

    np.random.seed(0)

    UM = UtilManager()

    if args.make_lightcone_dset:
        make_lightcone_dset(all_params, args)

    if args.plot_lightcones:
        plot_lightcones(all_params, args)
