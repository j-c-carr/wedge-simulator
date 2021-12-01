import os
import sys
import yaml
import h5py
import typing
from typing import Optional, List
import logging
import numpy as np
import py21cmfast as p21c
from random import randint
from pprint import pformat


def init_logger(f, name):
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")
    file_handler = logging.FileHandler(f)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


logger = init_logger("test.log", __name__)


class LightconeManager():
    """Manager class for creating 21cmFast Lightcones"""

    def __init__(self, 
                 config_params: dict) -> None:
        """
        Params:
        :config_params: Supplied when running python script, contains all
        information for creating lightcones
        """
        self.params = config_params

        self.astro_param_values = None
        if "astro_param_ranges" in self.params:
            self.generate_random_astroparams_from_range()
            
        # If AstroParams are supplied, must create the AstroParams object 
        if "astro_params" in self.params["p21c_run_lightcone_kwargs"]:
            self.params["p21c_run_lightcone_kwargs"]["astro_params"] = \
                p21c.AstroParams(**self.params["p21c_run_lightcone_kwargs"]["astro_params"])


    def random_from_range(self,
                          k: str) -> float:
        """Picks random number between range given by :k:, as specified in the
        config file."""

        assert k in self.params["astro_param_ranges"]

        l, r = self.params["astro_param_ranges"][k]

        # Return 3 floating point value in range [l, r]
        return np.random.randint(l*1000, high=r*1000) / 1000


    def generate_random_astroparams_from_range(self):
        """Generates astrophysical parameters from range for each lightcone"""
        self.astro_param_objects = []

        logger.debug("Generating {} astroparams...".format(self.params["num_lightcones"]))
        for i in range(self.params["num_lightcones"]):

            _astro_args = dict([(k, self.random_from_range(k)) for k in 
                    self.params["astro_param_ranges"].keys()])

            self.astro_param_objects.append(p21c.AstroParams(**_astro_args))

        # Format astro_param args to be saved to h5py file
        self.format_astro_params()


    def format_astro_params(self):
        """
        Re-formats AstroParam args so that they can be saved to h5py file as
        numpy arrays. A single np array will be saved to the h5py file for each
        astro_param. 
        Each entry in
            self.astro_param_values = {
                'HII_EFF_FACTOR': [30.0, 30.0, ..., 30.0],
                'F_STAR10': [-1.3, -1.3, ..., -1.3],
                ...
                'N_RSD_STEPS': [20, 20, ..., 20],
                },
        will be saved to the h5py file.
        """
        self.astro_param_values = {}

        # Save all AstroParam values:
        for k in self.astro_param_objects[0].__dict__.keys():

            if k == "INHOMO_RECO" or k == "_name":
                continue

            self.astro_param_values[k] = np.array([
                    self.astro_param_objects[i].__dict__[k] for i in
                            range(self.params["num_lightcones"])], dtype=np.float32)

            

    def generate_lightcones(self,
                            debug: bool = False) -> List[np.ndarray]:
        """
        Wrapper class for generating lightcones using the 21cmFAST 
        p21c.run_lightcones method. 
        -----
        Returns:
        :BT: (np.ndarray) Brightness temperature data from lightcones
        :XH: (np.ndarray) Ionization data from lightcones
        :lightcone_redshifts: (np.ndarray) redshifts values of the lightcones
        """

        BT = np.empty((self.params["num_lightcones"],
                    *self.params["original_lightcone_shape"]),
                    dtype=np.float32)
        XH = np.empty((self.params["num_lightcones"],
                    *self.params["original_lightcone_shape"]),
                    dtype=np.float32)

        lightcone_redshifts = np.empty(self.params["original_lightcone_shape"][0])

        # For some reason, normal random number generation doesn't work
        if "lightcone_random_seeds" not in self.params.keys():
            self.params["lightcone_random_seeds"] = \
                np.random.randint(1, high=9999, size=self.params["num_lightcones"])

        seed = self.params["lightcone_random_seeds"]

        logger.info(f"Generating lightcones...")
        for i in range(self.params["num_lightcones"]):

            logger.debug("p21c_run_lightcone_kwargs:")
            logger.debug(self.params["p21c_run_lightcone_kwargs"])

            # Generate AstroParams if specified in config file
            if hasattr(self, "astro_param_args"):
                logger.debug(self.astro_param_args[i])

                lightcone = p21c.run_lightcone(
                                **self.params["p21c_run_lightcone_kwargs"], 
                                astro_params = self.astro_param_objects[i],
                                random_seed = seed[i])

            else:
                lightcone = p21c.run_lightcone(
                                **self.params["p21c_run_lightcone_kwargs"], 
                                random_seed = seed[i])

            logger.debug("Lightcone dimensions: {}".format(
                         lightcone.lightcone_dimensions))

            # Make the LoS along the x axis
            bt = np.transpose(lightcone.brightness_temp, (2,1,0))
            xh = np.transpose(lightcone.xH_box, (2,1,0))

            # Assert the shapes match the config file
            assert bt.shape == self.params["original_lightcone_shape"], \
                    "expected {} but got {}".format(
                            self.params["original_lightcone_shape"], 
                            bt.shape)

            BT[i] = bt.astype(np.float32)
            XH[i] = xh.astype(np.float32)

            if i == 0:
                lightcone_redshifts = lightcone.lightcone_redshifts

            logger.info(f"Lightcone {i} done.")

        self.p21c_BT = BT
        self.p21c_XH = XH
        self.lightcone_redshifts = lightcone_redshifts


        return BT, lightcone_redshifts


