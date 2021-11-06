import typing
from typing import Union
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def blackman_harris_taper(x: np.ndarray) -> np.ndarray:
    """Applies blackman harris taper function along line of sight in real space"""
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    N = x.shape[0]

    bh = lambda n: a0 - \
            a1 * np.cos(2*np.pi * n/N) + \
            a2 * np.cos(4*np.pi * n/N) - \
            a3 * np.cos(6*np.pi * n/N)

    bh_window = np.array([bh(n) for n in range(N)])

    return x * bh_window.reshape(-1, 1, 1)


def compute_wedge_boundary(z: float, 
                           fov_angle: float = 90.,
                           O_m: float = 0.31,
                           O_d: float = 0.69) -> float:
    """Computes the wedge boundary at a given redshift and fov angle.
    Formula given in Mesinger et. al 2021 Eq (4)
    -----
    Params:
    :z: (float) redshift
    :fov_angle: angle of observation. Defaults to worst case at horizon
    :O_m: Normalized matter density
    :O_d: Normalized dark energy density
    """

    E = lambda z: np.sqrt(O_m * np.power(1+z, 3) + O_d)
    E_inv = lambda z: 1/(np.sqrt(O_m * np.power(1+z, 3) + O_d))

    a = E(z)
    b, _ = quad(E_inv, 0, z)

    return (np.sin(fov_angle*np.pi/180) * a * b) / (1+z)


def px_to_mpc(mode: int, N, mpc_res):
    """Converts fourier modes from px^{-1} to Mpc^{-1}"""
    assert N > 0 and mpc_res > 0
    return (2 * np.pi) / (N * mpc_res)


def pct_based_bar(x: np.ndarray,
        pct_range: float = 0.1) -> np.ndarray:
    """
    Applies blind bar to Fourier space.
    -----
    Parameters
    :x: (np.ndarray) 3-D fourier transformed lightcone. First axis is LoS.
    :pct_range: percentage of k_perp modes to remove. Default to 10%
    """

    # Remove the lowest k_parallel modes.
    assert pct_range < 1.0, \
        f"{pct_range} not in range (0,1)"

    dx = int(x.shape[0] * pct_range)
    mid = int(x.shape[0]/2)

    assert (mid - dx > 0) and (mid + dx < x.shape[0]), \
        f"dx {dx} with origin at {mid} is out of range"

    x[mid-dx:mid+dx] = 0j

    return x

def bar(x, maximum):
    """
    Applies blind bar to Fourier space.
    """
    DIM = np.shape(x)[0]
    half = int(DIM/2)
    DIMT = np.shape(x)[1]
    minimum = -1*maximum
    zeros = np.zeros((DIMT, DIMT)).astype(np.float16)
    for i in range(DIMT):
        if minimum<i-half<=maximum:
            x[i] = zeros
    return x


def sweep(x: np.ndarray, 
          z: float,
          fov_angle: float = 90.) -> np.ndarray:
    """
    Applies blind cones to Fourier space, taking advantage of 8-fold symmetery
    of the lightcone about the origin (center) of the box.
    ----------
    Parameters
    :x: fourier-transformed lightcone
    :z: redshift
    """
    assert x.shape[1] == x.shape[2], \
            f"expected square base but got shape x.shape[1:]"

    eps = 1e-18
    mid = x.shape[0]//2
    core = x.shape[1]//2

    wedge_boundary_slope = compute_wedge_boundary(z, fov_angle)

    wedge_angle = 90 - (np.arctan(wedge_boundary_slope) * 180 / np.pi)

    delta = x.shape[0] / x.shape[1]

    for i in range(mid):
        # Get threshold value for k_perp in Mpc
        k_parallel = i 
        radius = k_parallel / (wedge_boundary_slope + eps)

        for j in range(core):
            for k in range(core):

                k_perp = np.sqrt(j**2 + k**2)
                if (delta*k_perp) > radius:
                    x[mid+i,core+j,core+k] = 0j
                    x[mid-i-1,core+j,core+k] = 0j
                    x[mid+i,core-j-1,core+k] = 0j
                    x[mid-i-1,core-j-1,core+k] = 0j
                    x[mid+i,core+j,core-k-1] = 0j
                    x[mid-i-1,core+j,core-k-1] = 0j
                    x[mid+i,core-j-1,core-k-1] = 0j
                    x[mid-i-1,core-j-1,core-k-1] = 0j
    return x

