"""
Script for generating coeval boxes using py21cmfast
"""
import typing
from typing import Optional, List
import logging
import numpy as np
import py21cmfast as p21c

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


class CoevalManager():

    """Class for generating coeval boxes. """

    def __init__(self, ic_kwargs: Optional[dict] = None):
        """ic_kwargs: arguments to supply to initial conditions"""
        if ic_kwargs is None:
            self.ic_kwargs = {"user_params": {"HII_DIM": 128, "BOX_LEN": 128}}
        else:
            self.ic_kwargs = ic_kwargs


    def generate_coeval_boxes(self, redshifts: np.ndarray):
        """Generates coeval boxes at given redshifts using the initial
        conditions specified in the constructor"""

        logger.debug(f"Generating initial conditions...")
        initial_conditions = p21c.initial_conditions(**self.ic_kwargs)

        X = np.empty((redshifts.shape[0], 128, 128, 128), dtype=np.float32)

        logger.debug("Generating lightcones...")
        for i, z in enumerate(redshifts):

            coeval_box = p21c.run_coeval(
                    redshift = z,
                    init_box = initial_conditions)

            # Make the LoS along the x axis
            bt = np.transpose(coeval_box.brightness_temp, (2,1,0))

            # Assert the shapes match the config file
            assert bt.shape == (128, 128, 128), \
                    "expected {} but got {}".format(
                            (128, 128, 128), bt.shape)

            X[i] = bt.astype(np.float32)
            logger.debug(f"Coeval box {i} done.")

        return X

        
