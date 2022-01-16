"""
@author: j-c-carr
Manager class for generating py21cmFAST coeval boxes
"""

from typing import Optional
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


logger = init_logger(__name__, "wedge_simulator.log")


class CoevalManager:

    """
    Class for generating coeval boxes from p21cmFAST.
    Boxes are created from a single p21c.InitialConditions object.
    ----------
    Attributes:
    :ic_kwargs: Optional dictionary of kwargs to pass to the
                p21c.InitialConditions constructor.
    :box_shape: Shape of a single coeval box.
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
        self.box_shape = (ic_kwargs["user_params"]["HII_DIM"],
                          ic_kwargs["user_params"]["HII_DIM"],
                          ic_kwargs["user_params"]["HII_DIM"])

    def generate_coeval_boxes(self, redshifts: np.ndarray):
        """
        Generates coeval boxes at specified redshifts.
        ----------
        Params:
        :redshifts: Redshift of each coeval box
        ----------
        Returns:
        :BT: Brightness temperature data, TRANSPOSED so that the los axis is
             the first axis
        :XH: Ionization data, TRANSPOSED so that the los axis is the first axis
        """

        logger.debug(f"Generating initial conditions...")
        initial_conditions = p21c.initial_conditions(**self.ic_kwargs)

        BT = np.empty((redshifts.shape[0], *self.box_shape), dtype=np.float32)
        XH = np.empty((redshifts.shape[0], *self.box_shape), dtype=np.float32)

        logger.debug("Generating coeval boxes...")
        for i, z in enumerate(redshifts):

            coeval_box = p21c.run_coeval(redshift=z,
                                         init_box=initial_conditions)

            # Make the LoS along the first axis
            bt = np.transpose(coeval_box.brightness_temp, (2, 1, 0))
            xh = np.transpose(coeval_box.xH_box, (2, 1, 0))

            # Assert the shapes match the expected shape
            assert bt.shape == self.box_shape, \
                   "expected {} but got {}".format(self.box_shape, bt.shape)

            BT[i] = bt.astype(np.float32)
            XH[i] = xh.astype(np.float32)
            logger.debug(f"Coeval box {i} done.")

        return BT, XH

        
