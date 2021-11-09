"""
Script for manipulating coeval boxes in Fourier space (to be used in
fourier_manager.py)
"""
import typing
from typing import Union
import numpy as np
from scipy.integrate import quad


def compute_wedge_boundary(z: float, 
                           fov_angle: float = 90.,
                           O_m: float = 0.31,
                           O_d: float = 0.69) -> float:
    """
    Computes the wedge boundary at a given redshift and fov angle.
    Formula given in https://arxiv.org/pdf/1404.2596.pdf, Eq. (13)
    ----------
    Params:
    :z: (float) redshift
    :fov_angle: angle of observation. Defaults to worst case at horizon
    :O_m: Normalized matter density
    :O_d: Normalized dark energy density
    ----------
    Returns:
    Angle of the wedge region in cylindrical fourier space (float)
    """

    E = lambda z: np.sqrt(O_m * np.power(1+z, 3) + O_d)
    E_inv = lambda z: 1/(np.sqrt(O_m * np.power(1+z, 3) + O_d))

    a = E(z)
    b, _ = quad(E_inv, 0, z)

    return (np.sin(fov_angle*np.pi/180) * a * b) / (1+z)


def coeval_bar(x: np.ndarray,
        maximum: int) -> np.ndarray:
    """
    Removes smooth foreground contamination from 21cm coeval boxes
    ----------
    Params:
    :x: (np.ndarray) fourier-transformed coeval box.
    :maximum: (int) maximum fourier mode in k_parallel direction (corresponding
                    to an index along axis 0) to remove.
    ----------
    Returns:
    :x: (np.ndarray) fourier-transformed coeval box with smooth foregrounds
                     removed.
    """
    center = x.shape[0]//2
    assert center-maximum > 0 and center+maximum < x.shape[0],\
            "Number of foreground modes removed exceeds number of foreground modes"

    x[center-maximum:center+maximum] = 0

    return x


def coeval_sweep(x: np.ndarray, 
                 z: float) -> np.ndarray:
    """
    Removes the cylindrical wedge region in fourier space.
    ----------
    Params:
    :x: (np.ndarray) fourier-transformed coeval box.
    :z: (float) redshift for which to remove the wedge.
    ----------
    Returns:
    :x: (np.ndarray) fourier-transformed coeval box with wedge removed.
    """

    DIM1 = np.shape(x)[0]
    DIM2 = np.shape(x)[1]
    mid = int(DIM1/2)
    core = int(DIM2/2)

    wedge_boundary_slope = compute_wedge_boundary(z, 90.)

    alpha = 90 - (np.arctan(wedge_boundary_slope) * 180 / np.pi)

    for i in range(DIM1):
        x_distance = abs(mid-i)
        radius = np.tan(alpha*np.pi/180)*x_distance

        for j in range(DIM2):
            for k in range(DIM2):

                yz_distance = np.sqrt((j-core)**2+(k-core)**2)
                if yz_distance>radius:
                    x[i,j,k] = 0j

    return x

