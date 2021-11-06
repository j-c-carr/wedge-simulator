"""
by Samuel Gagnon-Hartman, 2019
modified by Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca), 2021

This code removes the foreground wedge for 21cm lightcones.
The transformations applied are performed in Fourier space and are meant to 
replicate the distortions and limitations of 'actual' datasets.

"""

import typing
from typing import List, Optional
import matplotlib.pyplot as plt
import logging
import numpy as np
from tqdm import tqdm
from filters import coeval_bar, coeval_sweep


def init_logger(f: str, 
                name: str):
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")
    file_handler = logging.FileHandler(f)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = init_logger("test.log", __name__)

class FourierManager():

    def fourier(self,
                x: np.ndarray) -> np.ndarray:
        """
        Applies a fourier transform to 3-D lightcone
        """
        return np.fft.fftshift(np.fft.fftn(x))


    def inverse(self, 
                x: np.ndarray) -> np.ndarray:
        """
        Applies inverse Fourier transform to return to 2D spatial map.
        """
        return np.fft.ifftn(np.fft.ifftshift(x))


    def remove_wedge(self, 
                     images: np.ndarray,
                     redshifts: np.ndarray) -> np.ndarray:
        """Performs wedge-removal in fourier space for each coeval box.
        images = coeval boxes generated from p21cmfast, masks = wedge-filtered
        boxes.
        """

        assert images.dtype == np.float32, \
            f"Images of dtype {images.dtype}, should be np.float32"

        assert images.shape[0] == redshifts.shape[0], \
            "Should be one redshift per image"

        masks = np.empty(images.shape)
        print(f"masks shape: {masks.shape}")
        for i, image in enumerate(images):

            transformed_image = np.empty(image.shape)

            # Apply Fourier transform to layer
            self.fourier_data = self.fourier(image)

            # Apply chosen transformations to layer
            coeval_bar(self.fourier_data, 2)
            coeval_sweep(self.fourier_data, redshifts[i])

            # Apply inverse fourier transform to layer
            transformed_image = np.real(self.inverse(self.fourier_data))

            # Convert to proper data type
            transformed_image = transformed_image.astype(np.float32)
            masks[i] = transformed_image

        return masks

