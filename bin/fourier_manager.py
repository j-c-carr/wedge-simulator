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
from filters import blackman_harris_taper, remove_wedge, old_bar, old_sweep, sweep
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


    def compute_difference(self, 
                           x1: np.ndarray, 
                           x2: np.ndarray) -> np.ndarray:
        """Computes amount of information lost between x1 and x2"""
        diff = x1 - x2
        return np.count_nonzero(diff) / diff.size


    def create_masks_legacy(self, 
                            images: np.ndarray,
                            redshifts: np.ndarray) -> np.ndarray:
        """LEGACY FUNCTION ----- USE create_masks() INSTEAD"""

        assert images.dtype == np.float32, \
            f"Images of dtype {images.dtype}, should be np.float32"

        masks = np.empty(images.shape)
        print(f"masks shape: {masks.shape}")
        for i, image in enumerate(images):

            transformed_image = np.empty(image.shape)
            # tmp = blackman_harris_taper(image)

            # Apply Fourier transform to layer
            self.fourier_data = self.fourier(image)


            # Apply chosen transformations to layer
            old_bar(self.fourier_data, 2)
            old_sweep(self.fourier_data, 8.5)
            #self.fourier_data = remove_wedge(self.fourier_data, redshifts[-1])


            # Apply inverse fourier transform to layer
            transformed_image = np.real(self.inverse(self.fourier_data))

            # Convert to proper data type
            transformed_image = transformed_image.astype(np.float32)
            masks[i] = transformed_image

        return images, masks, redshifts


    def create_masks_from_roll(self, 
                     images: np.ndarray, 
                     redshifts: np.ndarray,
                     mpc_per_px: float) -> List[np.ndarray]:
        """
        Applies wedge excision and noise filters to each lightcone in
        images. Sweep is the wedge removal and bar is the smooth foreground removal
        -----
        Params:
        :images: list of images (3D lightcones) to be transformed
        :redshifts: 1-D array of redshifts along the LoS
        :mpc_per_px: Mpc per pixel
        -----
        Returns:
        :images: list of original lightcones in the same range as masks
        :masks: list of Wedge-removed images
        """

        assert images.dtype == np.float32, \
            f"Images of dtype {images.dtype}, should be np.float32"
        assert redshifts.shape[0] == images.shape[1], \
            f"expected {images.shape[1]}, got {redshifts.shape[0]}"

        masks = np.empty(images.shape)
        print(f"masks shape: {masks.shape}")

        # Slide along 750 Mpc window
        window_length = 128 # int(750 / mpc_per_px)
        # print("Window length: ", window_length)

        start = np.where(np.floor(redshifts)==6.)[0][0]
        for i, image in enumerate(images):

            for z in tqdm(range(start-65, start+70)):

                dz = min(z+window_length, redshifts.shape[0]-1)
                tmp = np.empty((dz-z, *image.shape[1:]))

                tmp = image[z:dz]
                tmp = blackman_harris_taper(tmp)

                logger.debug(f"Removing wedge at redshift {redshifts[z]} from slice {z} to {dz}")
                self.fourier_data = self.fourier(tmp)
                
                # old_bar(self.fourier_data, 2)
                old_sweep(self.fourier_data, 15)
                # self.fourier_data = remove_wedge(self.fourier_data, redshifts[z])

                # No nan values
                assert np.isnan(np.sum(self.fourier_data)) == False

                # Inverse the transform and store the central slice
                tmp = np.real(self.inverse(self.fourier_data)).astype(np.float32)

                #if (z==0) or z==216:
                #    LPM.compare_lightcones(f"tmp_z_{z}"), \
                #            {"Image": image[:216].reshape(1, 216, 64, 64),
                #             "Mask": tmp.reshape(1, 216, 64, 64)}

                assert masks[i, z:dz].shape == tmp.shape, \
                    f"Expected {masks[i, z:dz].shape}, got {tmp.shape}"

                # Copy the middle slice
                masks[i, (z+dz)//2] = np.copy(tmp[(dz-z)//2])
                if dz == (image.shape[0]-1):
                    # LPM.compare_lightcones(f"tmp_z_{z}"), \
                    #         {"Image": image[:216].reshape(1, 216, 64, 64),
                    #          "Mask": tmp.reshape(1, 216, 64, 64)}
                    break

            logger.info(f"Image {i} done.")

        return images[:, start:start+128, ...], masks[:, start:start+128, ...],\
               redshifts[start:start+128]


    def create_masks(self, 
                     images: np.ndarray, 
                     redshifts: np.ndarray,
                     window_length: Optional[int] = 128) -> List[np.ndarray]:
        """
        Applies wedge excision and noise filters to each lightcone in
        images. Sweep is the wedge removal and bar is the smooth foreground removal
        -----
        Params:
        :images: list of images (3D lightcones) to be transformed
        :redshifts: 1-D array of redshifts along the LoS
        -----
        Returns:
        :images: list of original lightcones in the same range as masks
        :masks: list of Wedge-removed images
        """

        assert images.dtype == np.float32, \
            f"Images of dtype {images.dtype}, should be np.float32"
        assert redshifts.shape[0] == images.shape[1], \
            f"expected {images.shape[1]}, got {redshifts.shape[0]}"

        masks = np.zeros(images.shape)
        print(f"masks shape: {masks.shape}")

        print("Window length: ", window_length)

        dz = window_length // 2
        start = np.where(np.floor(redshifts)==7.)[0][0]

        assert start+256+dz < images.shape[1], \
                f"Lightcones of length {images.shape[1]}" \
                f"but filter requires length {start+256+dz}"
        # M = np.empty((256,))
        # j=0
        for i, image in enumerate(images):

            for z in tqdm(range(start, start+256)):
                tmp = np.empty((window_length, *image.shape[1:]))

                tmp = image[z-dz:z+dz]
                tmp = blackman_harris_taper(tmp)

                logger.debug(f"Removing wedge at redshift {redshifts[z]}" \
                             f"from slice {z} to {dz}")

                tmp_tilde = self.fourier(tmp)
                
                self.fourier_data = remove_wedge(tmp_tilde, redshifts[z])
                # print("Modes removed: ", m)

                # M[j] = m
                # j += 1

                # No nan values
                assert np.isnan(np.sum(self.fourier_data)) == False

                # Inverse the transform and store the central slice
                tmp = np.real(self.inverse(self.fourier_data)).astype(np.float32)

                if z==(-1):
                    diff = np.abs(image[z] - tmp[dz-1])
                    _d = diff > 1e-8
                    print("pct diff: ", _d.sum()/_d.size)
                    plt.imshow(diff, aspect="auto", origin="lower")
                    plt.savefig("cube_diff_64.png", dpi=400)

                # Copy the middle slice
                masks[i, z] = np.copy(tmp[dz-1])

            logger.info(f"Image {i} done.")

        # start = np.where(np.floor(redshifts)==6.)[0][0]
        return images[:, start:start+256, ...], masks[:, start:start+256, ...],\
               redshifts[start:start+256]



    def create_movie(self, 
                     images: np.ndarray, 
                     redshifts: np.ndarray,
                     window_length: Optional[int] = 128) -> List[np.ndarray]:
        """
        Applies wedge excision and noise filters to each lightcone in
        images. Sweep is the wedge removal and bar is the smooth foreground removal
        -----
        Params:
        :images: list of images (3D lightcones) to be transformed
        :redshifts: 1-D array of redshifts along the LoS
        -----
        Returns:
        :images: list of original lightcones in the same range as masks
        :masks: list of Wedge-removed images
        """


        assert images.dtype == np.float32, \
            f"Images of dtype {images.dtype}, should be np.float32"
        assert redshifts.shape[0] == images.shape[1], \
            f"expected {images.shape[1]}, got {redshifts.shape[0]}"

        masks = np.zeros((256, 256, 128))
        print(f"masks shape: {masks.shape}")

        print("Window length: ", window_length)

        dz = window_length // 2
        start = np.where(np.floor(redshifts)==7.)[0][0]

        assert start+256+dz < images.shape[1], \
                f"Lightcones of length {images.shape[1]}" \
                f"but filter requires length {start+256+dz}"
        # M = np.empty((256,))
        # j=0
        for i, image in enumerate(images):

            for z in tqdm(range(start, start+256)):
                if z > start:
                    masks[z-start] = np.copy(masks[z-start-1])
                else:
                    masks[z-start] = np.copy(image[start:start+256, :, 0])
                tmp = np.empty((window_length, *image.shape[1:]))

                tmp = image[z-dz:z+dz]
                tmp = blackman_harris_taper(tmp)

                logger.debug(f"Removing wedge at redshift {redshifts[z]}" \
                             f"from slice {z} to {dz}")

                tmp_tilde = self.fourier(tmp)
                
                self.fourier_data = remove_wedge(tmp_tilde, redshifts[z])
                # print("Modes removed: ", m)

                # M[j] = m
                # j += 1

                # No nan values
                assert np.isnan(np.sum(self.fourier_data)) == False

                # Inverse the transform and store the central slice
                tmp = np.real(self.inverse(self.fourier_data)).astype(np.float32)

                if False:
                    diff = np.abs(image[z] - tmp[dz-1])
                    _d = diff > 1e-8
                    print("pct diff: ", _d.sum()/_d.size)
                    plt.imshow(diff, aspect="auto", origin="lower")
                    plt.savefig("cube_diff_64.png", dpi=400)

                # Copy the middle slice
                masks[z-start, z-start] = np.copy(tmp[dz-1, 0])

            logger.info(f"Image {i} done.")
            break

        # start = np.where(np.floor(redshifts)==6.)[0][0]
        return masks, redshifts[start:start+256]

