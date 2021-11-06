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

        if "astro_param_distributions" in self.params:
            self.generate_random_astroparams()

        if "astro_param_ranges" in self.params:
            self.generate_random_astroparams_from_range()

        # Must create the AstroParams object if passed to run_lightcones
        if "astro_params" in self.params["p21c_run_lightcone_kwargs"]:
            self.params["p21c_run_lightcone_kwargs"]["astro_params"] = \
                p21c.AstroParams(**self.params["p21c_run_lightcone_kwargs"]["astro_params"])


    def random_sample(self, 
                      k: str) -> float:
        """
        Returns random sample from astro param distribution.
        -----
        Params:
        :k: AstroParam argument to sample from. Must be in config file
        """
        assert k in self.params["astro_param_distributions"]

        mu, std = self.params["astro_param_distributions"][k]

        return max(np.random.normal(mu, std), 1e-4)


    def generate_random_astroparams(self) -> None:
        """Generates astrophysical parameters from normal distribution"""

        self.astro_param_args = []

        # Generate a random set of AstroParameters for each sample
        logger.debug("Generating {} astroparams...".format(self.params["num_lightcones"]))
        for i in range(self.params["num_lightcones"]):

            _params = dict([(k, self.random_sample(k)) for k in 
                self.params["astro_param_distributions"].keys()])

            logger.debug(pformat(_params))
            self.astro_param_args.append(_params)

    
    def random_from_range(self,
                          k: str) -> float:
        """Picks random number between range given by :k:, as specified in the
        config file."""

        assert k in self.params["astro_param_ranges"]

        l, u = self.params["astro_param_ranges"][k]

        # Return 3 floating point value in range [l, u]
        return np.random.randint(l*1000, high=u*1000) / 1000


    def generate_random_astroparams_from_range(self) -> None:
        """Generates astrophysical parameters from range"""
        self.astro_param_args = []

        logger.debug("Generating {} astroparams...".format(self.params["num_lightcones"]))
        for i in range(self.params["num_lightcones"]):

            _params = dict([(k, self.random_from_range(k)) for k in 
                self.params["astro_param_ranges"].keys()])

            logger.debug(pformat(_params))
            self.astro_param_args.append(_params)
            

    def generate_lightcones(self,
                            debug: bool = False) -> List[np.ndarray]:
        """
        Wrapper class for generating lightcones using the 21cmFAST 
        p21c.run_lightcones method. 
        -----
        Returns:
        :X: (np.ndarray) Brightness temperature data from lightcones
        :lightcone_redshifts: (np.ndarray) redshifts values of the lightcones
        """

        X = np.empty((self.params["num_lightcones"],
                    *self.params["original_lightcone_shape"]),
                    dtype=np.float32)

        lightcone_redshifts = np.empty(self.params["original_lightcone_shape"][0])

        # For some reason, normal random number generation doesn't work
        if "lightcone_random_seeds" not in self.params.keys():
            self.params["lightcone_random_seeds"] = \
                np.random.randint(1, high=9999, size=self.params["num_lightcones"])

        seed = self.params["lightcone_random_seeds"]

        logger.debug(f"Generating lightcones...")
        for i in range(self.params["num_lightcones"]):

            # Generate AstroParams if specified in config file
            if hasattr(self, "astro_param_args"):
                self.params["p21c_run_lightcone_kwargs"]["astro_params"] = \
                        p21c.AstroParams(**self.astro_param_args[i])

            logger.debug("p21c_run_lightcone_kwargs:")
            logger.debug(pformat(self.params["p21c_run_lightcone_kwargs"]))

            lightcone = p21c.run_lightcone(
                    **self.params["p21c_run_lightcone_kwargs"],
                    random_seed = seed[i],
                    direc="scratch/__tmp")
            
            logger.debug("Lightcone dimensions: ",
                    lightcone.lightcone_dimensions)

            # Make the LoS along the x axis
            bt = np.transpose(lightcone.brightness_temp,(2,1,0))

            # Assert the shapes match the config file
            assert bt.shape == self.params["original_lightcone_shape"], \
                    "expected {} but got {}".format(
                            self.params["original_lightcone_shape"], 
                            bt.shape)

            X[i] = bt.astype(np.float32)

            if i == 0:
                lightcone_redshifts = lightcone.lightcone_redshifts

            logger.info(f"Lightcone {i} done.")


        return X, lightcone_redshifts


