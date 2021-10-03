"""
Author: Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca)

Generates a data set using 21cmFAST and wedge removal from fourier.py
Data set is saved as a single h5 file
"""

import os
import sys
import h5py
import yaml
import typing
from typing import Optional
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from lightcone_plot_manager import LightconePlotManager
from lightcone_manager import LightconeManager
from fourier_manager import FourierManager

# Prints all logging info to std.err
logging.getLogger().addHandler(logging.StreamHandler())

plt.style.use("lightcones/plot_styles.mplstyle")
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
print(COLORS)


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


def make_coeval_boxes(all_params: dict, 
               args: argparse.ArgumentParser) -> None:
    """Generates lightcones and masks for each set of initial conditions"""


    with h5py.File(f"scratch/datasets/{args.name}.h5", "w") as hf:

        for dset_name, initial_conditions in all_params["lightcone_params"].items():

            LM = LightconeManager(initial_conditions)
            FM = FourierManager()

            Y, redshifts = LM.generate_coeval_boxes()
            X = FM.create_masks_legacy(Y)

            #LPM = LightconePlotManager(redshifts, (512, 64, 64),
            #                           (2080, 220, 220))

            #LPM.compare_lightcones(f"{dset_name}_fast_sweep", \
            #                       {"Image": images, "Mask": masks})

            hf.create_dataset(f"coeval_inputs", data=X)
            hf.create_dataset(f"coeval_targets", data=Y)
            hf.create_dataset(f"coeval_redshifts", data=redshifts)

            # Store extra data 
            if hasattr(LM, "metadata"):
                for k in LM.metadata.keys():
                    hf.create_dataset(f"{k}",
                            data=LM.metadata[k])
                    logger.debug(f"\n----\tSaved {k}\t----\n")
                    logger.debug(LM.metadata[k])
                    
        save_dataset_attrs(hf, all_params)


def plot_coeval_boxes(all_params: dict,
               args: argparse.ArgumentParser) -> None:
    """Plot samples from the dataset"""

    with h5py.File(f"scratch/datasets/{args.name}.h5", "r") as hf:

        for dset_name, initial_conditions in all_params["lightcone_params"].items():

            print(list(hf.keys()))
            inputs = hf[f"coeval_inputs"][:]
            targets = hf[f"coeval_targets"][:]
            redshifts = hf[f"coeval_redshifts"][:]
            redshifts = np.linspace(7, 9, inputs.shape[1])
            LPM = LightconePlotManager(redshifts, (128, 128, 128), (256, 256, 256))

            LPM.compare_lightcones(f"{dset_name}_coeval", 
                    {"inputs": inputs, "targets": targets},
                    num_samples=1)


def test_cubes(all_params: dict, 
               args: argparse.ArgumentParser) -> None:

    for dset_name, initial_conditions in all_params["lightcone_params"].items():

        LM = LightconeManager(initial_conditions)
        FM = FourierManager()

        lightcones, redshifts = LM.generate_lightcones()

        # For blocks
        # start = np.where(np.floor(redshifts)==6.)[0][0]
        # cubes = lightcones[:, start: start+128]
        # print("cut at redshift: ", redshifts[start+128])
        # mask_cubes = FM.create_masks_legacy(cubes, redshifts[start: start+128])

        # For roll
        start = np.where(np.floor(redshifts)==6.)[0][0]
        images, masks, redshifts = FM.create_masks(lightcones, redshifts, 2)
        print("redshifts shape: ", redshifts.shape)
        print("images shape: ", images.shape)
        print("masks shape: ", masks.shape)
        LPM = LightconePlotManager(redshifts, (512, 128, 128),
                (2840, 512, 512))

        LPM.compare_lightcones(f"{dset_name}_clean", \
                               {"LC": images, 
                                "LC - Wedge": masks})


def modify_cones(all_params: dict, 
               args: argparse.ArgumentParser) -> None:
    """Generates lightcones and masks for each set of initial conditions"""


    with h5py.File(f"scratch/datasets/{args.name}.h5", "r+") as hf:

        for dset_name, initial_conditions in all_params["lightcone_params"].items():

            LM = LightconeManager(initial_conditions)
            FM = FourierManager()

            lightcones, redshifts = LM.generate_lightcones()

            # Perturb the means
            means = lightcones.mean(axis=(2,3), keepdims=True)
            middle = means.mean()
            add_back = np.ones(lightcones.shape) * middle

            lightcones -= means
            lightcones += add_back

            images, masks, redshifts = FM.create_masks(lightcones, redshifts)

            
            # diff = masks_new128 - masks_new256
            # lightcones = lightcones[:, :256, :128, :128]

            # LPM = LightconePlotManager(redshifts[:256], (256, 128, 128),
            #                            (512, 256, 256))
            # LPM.compare_lightcones("table_test", 
            #                        {"lightcones": lightcones,
            #                         "wedge-removed lightcones": lightcones,
            #                         "predictions": lightcones})

            hf.create_dataset(f"{dset_name}_images", data=images)
            hf.create_dataset(f"{dset_name}_masks", data=masks)
            hf.create_dataset(f"{dset_name}_redshifts", data=redshifts)

            # Store extra data 
            if hasattr(LM, "metadata"):
                for k in LM.metadata.keys():
                    hf.create_dataset(f"{k}",
                            data=LM.metadata[k])
                    logger.debug(f"\n----\tSaved {k}\t----\n")
                    logger.debug(LM.metadata[k])
                    
        save_dataset_attrs(hf, all_params)


def make_cones(all_params: dict, 
               args: argparse.ArgumentParser) -> None:
    """Generates lightcones and masks for each set of initial conditions"""


    with h5py.File(f"scratch/datasets/{args.name}.h5", "w") as hf:

        for dset_name, initial_conditions in all_params["lightcone_params"].items():

            LM = LightconeManager(initial_conditions)
            FM = FourierManager()

            lightcones, redshifts = LM.generate_lightcones()

            # Perturb the means
            means = lightcones.mean(axis=(2, 3), keepdims=True)
            a = np.ones(lightcones.shape) * means.mean()
            lightcones -= means
            lightcones += a

            # start = np.where(np.floor(redshifts)==7.)[0][0]
            # cones = lightcones[:, start:start+256]
            # shifts = redshifts[start:start+256]

            # For roll
            # _, masks_legacy, __ = FM.create_masks_legacy(cones, shifts)
            # _, masks128, __ = FM.create_masks(lightcones, redshifts, 2, window_length=128)
            images, masks, redshifts = FM.create_masks(lightcones, redshifts)

            
            # diff = masks_new128 - masks_new256
            # lightcones = lightcones[:, :256, :128, :128]

            # LPM = LightconePlotManager(redshifts[:256], (256, 128, 128),
            #                            (512, 256, 256))
            # LPM.compare_lightcones("table_test", 
            #                        {"lightcones": lightcones,
            #                         "wedge-removed lightcones": lightcones,
            #                         "predictions": lightcones})

            hf.create_dataset(f"{dset_name}_images", data=images)
            hf.create_dataset(f"{dset_name}_masks", data=masks)
            hf.create_dataset(f"{dset_name}_redshifts", data=redshifts)

            # Store extra data 
            if hasattr(LM, "metadata"):
                for k in LM.metadata.keys():
                    hf.create_dataset(f"{k}",
                            data=LM.metadata[k])
                    logger.debug(f"\n----\tSaved {k}\t----\n")
                    logger.debug(LM.metadata[k])
                    
        save_dataset_attrs(hf, all_params)


def create_movie(all_params: dict, 
               args: argparse.ArgumentParser) -> None:
    """Generates lightcones and masks for each set of initial conditions"""


    with h5py.File(f"scratch/datasets/{args.name}.h5", "w") as hf:

        for dset_name, initial_conditions in all_params["lightcone_params"].items():

            LM = LightconeManager(initial_conditions)
            FM = FourierManager()

            lightcones, redshifts = LM.generate_lightcones()

            masks_slices, redshifts = FM.create_movie(lightcones, redshifts)
            # redshifts = redshifts[0:256]
            # masks_slices = np.random.randn(256, 256, 128)
            LPM = LightconePlotManager(redshifts, (256, 128, 128), (512, 256, 256))

            LPM.lightcone_movie(masks_slices, redshifts)


def plot_pixel_dist(X):
    """Plots min/max pixels per redshift"""
    assert X.ndim == 4, \
            f"expected 4 dims but got shape {X.shape}"

    for lc in X:
        mins = []
        maxes = []
        # Find min/max pixels along LoS
        for i in range(lc.shape[0]):
            mins.append(lc[i].min())
            maxes.append(lc[i].max())
            plt.vlines(i, mins[i], maxes[i])
    plt.savefig("min_max_half.png", dpi=400)


def plot_cmaps(all_params: dict,
               args: argparse.ArgumentParser) -> None:
    """Plot samples from the dataset"""

    with h5py.File("scratch/results/z7-9_HII-DIM-128_BOX-LEN-256_normed-params-and-input_beta001_gillet_192-boxes_validation.h5", "r") as hf:
        print(list(hf.keys()))
        for dset_name, initial_conditions in all_params["lightcone_params"].items():

            images = hf[f"X"][:]
            masks = hf[f"Ytrue"][:]
            predictions = hf[f"Ypred"][:]
            redshifts = hf[f"redshifts"][:]

            masks = masks.reshape(-1, 256, 128, 128)
            images = images.reshape(-1, 256, 128, 128)
            predictions = predictions.reshape(-1, 256, 128, 128)

            LPM = LightconePlotManager(redshifts[:256], (256, 128, 128),
                                       (512, 256, 256))
            LPM.compare_lightcones("cmap_test", 
                                   {"lightcones": images,
                                    "wedge-removed lightcones": masks,
                                    "predictions": predictions})
            
            LPM = LightconePlotManager(redshifts, (512, 64, 64),
                                       (2048, 256, 256))

            LPM.compare_lightcones(f"gillet_collwarm", 
                    {"images": images, "masks": masks}, num_samples=10)


def plot_cones(all_params: dict,
               args: argparse.ArgumentParser) -> None:
    """Plot samples from the dataset"""

    with h5py.File(f"scratch/datasets/{args.name}.h5", "r") as hf:


        for dset_name, initial_conditions in all_params["lightcone_params"].items():

            images = np.array([hf[k][:] for k in hf.keys() if "masks" in k],
                          dtype=np.float32)
            masks = np.array([hf[k][:] for k in hf.keys() if "images" in k], 
                          dtype=np.float32)
            redshifts = np.array([hf[k][:] for k in hf.keys() if "redshifts" in k], 
                          dtype=np.float32)
            # images = hf[f"{dset_name}_images"][:]
            # masks = hf[f"{dset_name}_masks"][:]
            # redshifts = hf[f"{dset_name}_redshifts"][:]

            masks = masks.reshape(-1, 256, 128, 128)
            images = images.reshape(-1, 256, 128, 128)
            redshifts = redshifts.reshape(-1)
            print("data shape: ", masks.shape)

            ##########################
            #  Lightcone mean plots  #
            ##########################
            
            # Mean of all lightcones
            all_stds = masks.std(axis=(2,3)) ** 2
            stds = masks.std(axis=(0, 2,3)) ** 2
            amax = all_stds.max(axis=(0))
            amin = all_stds.min(axis=(0))
            print("Means shape: ", stds.shape)
            print("Amax shape: ", amax.shape)
            print("Amin shape: ", amin.shape)
            print("Redshifts shape: ", redshifts.shape)

            # # Replace with random sample
            lc1 = masks[8]
            lc1_std = lc1.std(axis=(1,2)) **2

            # lc2 = images[13]
            # lc2_mean = lc2.mean(axis=(1,2))

            plt.plot(redshifts, stds, "-", lw=3, label="Average")
            plt.fill_between(redshifts, amax, amin, alpha=0.3)
            plt.scatter(redshifts, lc1_std, color="r", s=3, label="LC 8")
            #plt.scatter(redshifts, lc2_mean, s=3, label="LC 13")
            plt.xlabel(r"$z$")
            plt.ylabel("$\sigma^{2}$")
            plt.title("Variance for Lightcones (WR-LC)")
            plt.legend()
            plt.savefig("freq_slice_stds_for_images.png", dpi=400)

            ###############################
            #  Lightcone pixel val plots  #
            ###############################

            # Mean of all lightcones
            #all_means = images.mean(axis=(2,3))
            #means = images.mean(axis=(0, 2,3))
            #amax = all_means.max(axis=(0))
            #amin = all_means.min(axis=(0))
            #print("Means shape: ", means.shape)
            #print("Amax shape: ", amax.shape)
            #print("Amin shape: ", amin.shape)
            #print("Redshifts shape: ", redshifts.shape)

            # # Replace with random sample
            # lc1 = images[8]
            # lc1_mean = lc1.mean(axis=(1,2))
            # lc1_max = lc1.max(axis=(1,2))
            # lc1_min = lc1.min(axis=(1,2))

            # lc2 = images[13]
            # lc2_mean = lc2.mean(axis=(1,2))

            # plt.plot(redshifts, lc1_mean, "-", lw=2, label="Average")
            # plt.fill_between(redshifts, lc1_max, lc1_min, alpha=0.3)
            # #plt.scatter(redshifts, lc1_mean, color="r", s=3, label="LC 8")
            # #plt.scatter(redshifts, lc2_mean, s=3, label="LC 13")
            # plt.xlabel(r"$z$")
            # plt.ylabel("Pixel Value")
            # plt.title("Pixel Value Range for a Single Lightcone")
            # plt.legend()
            # plt.savefig("freq_slice_range_for_mask.png", dpi=400)

            ###############
            #  Histogram  #
            ###############
            #lc1 = images[13]
            #X1 = lc1[0, :, :].flatten()
            #plt.hist(X1, bins=20, alpha=0.8, label="$z=7$")
            #plt.title(r"Histogram at $z=7$")
            ##plt.savefig("lc_slice_0.png")
            #X2 = lc1[-1, :, :].flatten()
            #plt.hist(X2, bins=20, alpha=0.8, label="$z=8.75$")
            #plt.title(r"Distribution of Pixels at $z=7$ and $z=9$")
            #plt.legend()
            #plt.savefig("constmean_lc_13_slices_0_255.png")


def parse_args():
    """Handle the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="name of dataset. The dataset will be saved to scratch/datasets/<name>.h5")
    parser.add_argument("config_file", help="filepath to .yml configuration file")
    parser.add_argument("-mc", "--make_cones", action="store_true", help="generate lightcones")
    parser.add_argument("-pc", "--plot_cones", action="store_true", help="plot the dataset")
    parser.add_argument("--modify_cones", action="store_true", help="modify the dataset")
    parser.add_argument("-mcb", "--make_coeval_boxes", action="store_true",
            help="generate coeval boxes")
    parser.add_argument("-pcb", "--plot_coeval_boxes", action="store_true",
            help="plot coeval boxes")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    
    # Handle the command line arguments
    args = parse_args()
    all_params = read_params_from_yml_file(args.config_file)
    logger = init_logger("test.log", __name__)

    if args.make_cones:
        make_cones(all_params, args)

    if args.modify_cones:
        modify_cones(all_params, args)

    if args.make_coeval_boxes:
        make_coeval_boxes(all_params, args)

    if args.plot_coeval_boxes:
        plot_coeval_boxes(all_params, args)

    if args.modify_cones:
        plot_cmap(all_params, args)

    if args.plot_cones:
        logger.info("Plotting lightcones...")
        plot_cones(all_params, args)
        logger.info("Done.")

