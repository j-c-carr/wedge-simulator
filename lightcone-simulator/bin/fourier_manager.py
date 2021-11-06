"""
Author: Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca), 2021

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
from filters import blackman_harris_taper, bar, sweep
from lightcone_plot_manager import LightconePlotManager


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
    """
    Wrapper class for removing the wedge of a 21cmFAST coeval box in fourier
    space. The wedge region that is removed is described in
    https://arxiv.org/pdf/1404.2596.pdf (Liu et al., 2014)
    """

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


    def remove_wedge_from_lightcones(self, 
                                     lightcones: np.ndarray, 
                                     redshifts: np.ndarray,
                                     starting_redshift: float,
                                     n_los_pixels: int) -> List[np.ndarray]:
        """
        Performs wedge-removal in fourier space for each lightcone as desrcibed
        in Prelogovic et al, https://arxiv.org/abs/2107.00018.
        ----------
        Params:
        :lightcones: lightcone brightness temperature boxes generated from 
                     21cmFAST.
        :redshifts: 1-D array of redshifts along the LoS.
        :starting_redshift: determines redshift of the first slice of the
                            lightcone.
        :n_los_pixels: number of pixels along the line of sight to be kept.
        ----------
        Returns:
        :lightcones: truncated version of the original lightcones that 
                     match the starting redshift and n_los_pixels.
        :wedge_filtered_lightcones: list of wedge-removed lightcones
        :redshifts: redshift values for each pixel along the los.
        """

        assert lightcones.dtype == np.float32, \
            f"Lightcones of dtype {lightcones.dtype}, should be np.float32"

        assert redshifts.shape[0] == lightcones.shape[1], \
            f"expected {lightcones.shape[1]}, got {redshifts.shape[0]}"

        wedge_removed_lightcones = np.zeros(lightcones.shape)

        # rolling window length should match length scales in the other
        # directions.
        window_length = lightcones.shape[-1]
        dz = window_length // 2
        start = np.where(np.floor(redshifts)==starting_redshift)[0][0]

        # Make sure filtering algorithm has a large enough inital box
        assert start-dz >= 0, \
                f"Filter requires starting_redshift to be (at least)" \
                f"at index {dz}, but starting_redshift is at {start}"

        assert start+n_los_pixels+dz < lightcones.shape[1], \
                f"Lightcones of length {lightcones.shape[1]}" \
                f"but filter requires length {start+n_los_pixels+dz}"


        for i, lightcone in enumerate(lightcones):

            for z in tqdm(range(start, start+n_los_pixels)):
                tmp = np.empty((window_length, *lightcone.shape[1:]))

                tmp = lightcone[z-dz:z+dz]
                tmp = blackman_harris_taper(tmp)

                logger.debug(f"Removing wedge at redshift {redshifts[z]}" \
                             f"from slice {z} to {dz}")

                tmp_tilde = self.fourier(tmp)
                tmp_tilde = bar(tmp_tilde, 2)
                self.fourier_data = sweep(tmp_tilde, redshifts[z])

                # No nan values
                assert np.isnan(np.sum(self.fourier_data)) == False

                # Inverse the transform and store the central slice
                tmp = np.real(self.inverse(self.fourier_data)).astype(np.float32)

                # Copy the middle slice only
                wedge_removed_lightcones[i, z] = np.copy(tmp[dz-1])

            logger.info(f"Image {i} done.")

        # Return the desired lightcone range
        return lightcones[:, start:start+n_los_pixels, ...],\
               wedge_removed_lightcones[:, start:start+n_los_pixels, ...],\
               redshifts[start:start+n_los_pixels]



