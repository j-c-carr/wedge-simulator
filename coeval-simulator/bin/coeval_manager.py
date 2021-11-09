"""
Script for generating coeval boxes using py21cmfast
"""
import typing
from typing import Optional, List
import logging
import numpy as np
import py21cmfast as p21c

def init_logger(name, f: Optional[str] = None):
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")
    if f is not None:
        file_handler = logging.FileHandler(f)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = init_logger(__name__, "test.log")


class CoevalManager():

    """
    Class for generating coeval boxes from p21cmFAST.
    Boxes are created from a single p21c.InitialConditions object, with gets
    instantiated based on the parameters supplied by the user in the config
    file.
    """

    def __init__(self, ic_kwargs: Optional[dict] = None):
        """ic_kwargs: arguments to supply to p21c.initial_conditions"""

        if ic_kwargs is None:
            ic_kwargs = {"user_params": {"HII_DIM": 128, "BOX_LEN": 128},
                         "random_seed": 0}

        assert "HII_DIM" in ic_kwargs["user_params"] and \
               "BOX_LEN" in ic_kwargs["user_params"] and \
               "random_seed" in ic_kwargs, \
               "Must supply BOX_LEN, HII_DIM and random_seed" 

        self.ic_kwargs = ic_kwargs
        self.box_shape = (ic_kwargs["user_params"]["BOX_LEN"],
                          ic_kwargs["user_params"]["BOX_LEN"],
                          ic_kwargs["user_params"]["BOX_LEN"])


    def generate_coeval_boxes(self, redshifts: np.ndarray):
        """
        Generates coeval boxes at given redshifts using the initial
        condition parameters specified in the constructor.
        ----------
        Params:
        :redshifts: (np.ndarray) redshift of each coeval box
        ----------
        Returns:
        :BT: (np.ndarray) brightness temperature data, TRANSPOSED so that the
                          los axis is the first axis
        :XH: (np.ndarray) ionization data, TRANSPOSED so that the los axis is
                          the first axis
        """

        logger.debug(f"Generating initial conditions...")
        initial_conditions = p21c.initial_conditions(**self.ic_kwargs)

        
        BT = np.empty((redshifts.shape[0], *self.box_shape), dtype=np.float32)
        XH = np.empty((redshifts.shape[0], *self.box_shape), dtype=np.float32)

        logger.debug("Generating coeval boxes...")
        for i, z in enumerate(redshifts):

            coeval_box = p21c.run_coeval(
                    redshift = z,
                    init_box = initial_conditions)

            # Make the LoS along the first axis
            bt = np.transpose(coeval_box.brightness_temp, (2,1,0))
            xh = np.transpose(coeval_box.xH_box, (2,1,0))

            # Assert the shapes match the expected shape
            assert bt.shape == self.box_shape, \
                    "expected {} but got {}".format(
                            self.box_shape, bt.shape)

            BT[i] = bt.astype(np.float32)
            XH[i] = xh.astype(np.float32)
            logger.debug(f"Coeval box {i} done.")

        return BT, XH

        
