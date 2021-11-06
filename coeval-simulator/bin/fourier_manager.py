"""
by Samuel Gagnon-Hartman, 2019
modified by Jonathan Colaco Carr (jonathan.colacocarr@mail.mcgill.ca), 2021

This code removes the foreground wedge for 21cm lightcones.
The transformations applied are performed in Fourier space and are meant to 
replicate the distortions and limitations of 'actual' datasets.

"""
import logging
import numpy as np
import typing
from typing import List, Optional
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


    def remove_wedge(self, 
                     boxes: np.ndarray,
                     redshifts: np.ndarray) -> np.ndarray:
        """
        Performs wedge-removal in fourier space for each coeval box. The
        wedge-shaped region that is removed is formalised in 
        https://arxiv.org/pdf/1404.2596.pdf. The implementation of
        wedge-removal is described in Gagnon-Hartmen et al.,
        https://arxiv.org/pdf/2102.08382.pdf
        ----------
        Params:
        :boxes: Coeval boxes brightness temperature data generated from 
                21cmFAST
        :redshifts: Redshifts corresponding to each coeval box
        ----------
        Returns:
        :wedge_filtered_boxes: (np.ndarray) coeval boxes with the wedge-region
                               removed.
        """

        assert boxes.dtype == np.float32, \
            f"Boxes of dtype {boxes.dtype}, should be np.float32"

        assert boxes.shape[0] == redshifts.shape[0], \
            "Should be one redshift per box"

        wedge_filtered_boxes = np.empty(boxes.shape)
        for i, box in enumerate(boxes):

            transformed_box = np.empty(box.shape)

            self.fourier_data = self.fourier(box)

            # coeval_bar removes smooth foregrounds
            coeval_bar(self.fourier_data, 2)
            # coeval_sweep removes wedge-shaped region
            coeval_sweep(self.fourier_data, redshifts[i])

            # Apply inverse fourier transform to layer
            transformed_box = np.real(self.inverse(self.fourier_data))

            transformed_box = transformed_box.astype(np.float32)
            wedge_filtered_boxes[i] = transformed_box

        return wedge_filtered_boxes

