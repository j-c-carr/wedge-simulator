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


def bar(x: np.ndarray,
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


def remove_wedge(x: np.ndarray, 
                 z: float) -> np.ndarray:
    """Removes fourier modes inside foreground wedge"""
    x1 = old_bar(x, 2)
    # tmp = np.copy(x1)
    x2 = sweep(x1, z)

    # assert tmp.shape == x2.shape
    # removed = x2[(x2.shape[0]//2)-1] - tmp[(x1.shape[0]//2)-1]
    # pct_modes_removed = 100 * (1 - (np.count_nonzero(removed)/removed.size))

    return x
    
 
def coeval_bar(x, maximum):
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


def coeval_sweep(x, z, fill=False):
    """
    Applies blind cones to Fourier space.
    -alpha: the external angle of the cone w.r.t. the z=0 plane
    -fill: fills wedge with Gaussian noise if True
    """
    DIM1 = np.shape(x)[0]
    DIM2 = np.shape(x)[1]
    mid = int(DIM1/2)
    core = int(DIM2/2)

    count=0

    wedge_boundary_slope = compute_wedge_boundary(z, 90.)

    wedge_angle = 90 - (np.arctan(wedge_boundary_slope) * 180 / np.pi)
    print("wedge angle: ", wedge_angle)
    alpha = wedge_angle

    for i in range(DIM1):
        x_distance = abs(mid-i)
        radius = np.tan(alpha*np.pi/180)*x_distance
        for j in range(DIM2):
            for k in range(DIM2):
                yz_distance = np.sqrt((j-core)**2+(k-core)**2)
                if yz_distance>radius:
                    x[i,j,k] = 0j
                    count+=1
    print("Modes removed: ", 100*count/x.size)
    return x

