"""
Author: Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca)

Generates a data set using 21cmFAST and wedge removal from fourier.py
Data set is saved as a single h5 file
"""

import h5py
import yaml
from pprint import pprint, pformat
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
    

def modify_lightcones(all_params: dict, 
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


def save_dset_to_hf(filename: str,
                    data: dict,
                    attrs: dict = {},
                    astro_param_values: Optional[dict] = None,
                    astro_param_ranges: Optional[dict] = None):

    with h5py.File(f"{args.dset_dir}/{args.dset_name}.h5", "w") as hf:

        # Save datasets
        for k, v in data.items():
            hf.create_dataset(k, data=v)

        # Save attributes
        for k, v in attrs.items():
            hf.attrs[k] = str(v)

        # Save astro physical parameters, if available
        if astro_param_values is not None:

            astro_params = hf.create_group("astro_params")

            # Save the ranges of random astro params
            if astro_param_ranges is not None:
                astro_params.attrs["astro_param_ranges"] = \
                        str(astro_param_ranges)

            # Save the values of all astro params
            for k, v in astro_param_values.items():
                astro_params.create_dataset(k, data=v)

    # On success
    logger.info("\n----------\n")
    logger.info(f"h5py file created at {filename}")
    logger.info("Datasets:")
    for k in data.keys():
        logger.info("\t'{}', shape: {}".format(k, data[k].shape))
    logger.info("Attributes:")
    for k in attrs.keys():
        logger.info("\t'{}': {}".format(k, attrs[k]))

    if astro_param_values is not None:
        logger.info("AstroParams: ")
        for k in astro_param_values.keys():
            logger.info("\t'{}': {}".format(k, astro_param_values[k]))
    logger.info("\n----------\n")


def make_lightcone_dset(all_params: dict, 
                        args: argparse.ArgumentParser) -> None:
    """
    Generates a dataset of lightcones and associated wedge-filtered lightcones,
    saving it to an h5 file.
    ----------
    Params:
    :all_params: (dict) All the parameters specified in config file
    """


    LM = LightconeManager(all_params)
    FM = FourierManager()

    p21c_lightcones, redshifts = LM.generate_lightcones()

    # Remove the mean along each frequency slice to simulate observations
    p21c_lightcones -= p21c_lightcones.mean(axis=(2, 3), keepdims=True)

    starting_redshift = all_params["final_starting_redshift"]
    start = np.where(np.floor(redshifts)==starting_redshift)[0][0]
    n_los_pixels = all_params["final_lightcone_shape"][0]

    xh_boxes = LM.p21c_XH[:, start:start+n_los_pixels]

    mpc_res = all_params["final_lightcone_dimensions"][0] / \
              all_params["final_lightcone_shape"][0]

    lightcones, wedge_filtered_lightcones, redshifts = \
            FM.remove_wedge_from_lightcones(p21c_lightcones, 
                                                  redshifts,
                                                  starting_redshift,
                                                  n_los_pixels,
                                                  mpc_res)

    assert xh_boxes.shape == lightcones.shape, \
            "shape of ionized boxes and lightcones match but got shapes" +\
            f"{xh_boxes.shape} and {lightcones.shape}"

    if hasattr(LM, "astro_param_values"):
        astro_param_values = LM.astro_param_values
    else:
        astro_param_values = None

    save_dset_to_hf(f"{args.dset_dir}/{args.dset_name}.h5",
                    {"random_seeds": LM.params["lightcone_random_seeds"], 
                     "redshifts": redshifts,
                     "lightcones": lightcones,
                     "wedge_filtered_lightcones": wedge_filtered_lightcones,
                     "ionized_boxes": xh_boxes},
                    attrs = {"p21c_run_lightcone_kwargs": \
                             str(all_params["p21c_run_lightcone_kwargs"])},
                    astro_param_values = astro_param_values)
        

def parse_args():
    """Handle the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("dset_dir", help="parent directory of dataset")
    parser.add_argument("dset_name", help="name of dataset. The dataset will be saved to <dset_dir>/<dset_name>.h5")
    parser.add_argument("config_file", help="filepath to .yml configuration file")
    parser.add_argument("--make_lightcone_dset", action="store_true", help="generate lightcones")
    parser.add_argument("--modify_lightcones", action="store_true", help="modify lightcones")
    parser.add_argument("--old_data", help="name of an existing dataset (only used for --modify_lightcones")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    
    # Handle the command line arguments
    args = parse_args()
    all_params = read_params_from_yml_file(args.config_file)
    logger = init_logger("test.log", __name__)

    #np.random.seed(0)

    UM = UtilManager()

    if args.make_lightcone_dset:
        make_lightcone_dset(all_params, args)

    if args.plot_lightcones:
        modify_lightcones(all_params, args)

