"""
Operations in Fourier space to remove wedge modes of simulated py21cmFAST
lightcones.
@author: j-c-carr

"""

import numpy as np
from scipy.integrate import quad

# Physical constants
O_m = 0.31                  # Normalized matter density
O_d = 0.69                  # Normalized dark matter density
H_0 = 69.7e3                # Hubble constant ((m/s) / Mpc)
NU_21 = 1420                # Frequency of 21cm signal (MHz)
LIGHTSPEED = 299792458      # Speed of light constant (m/s)

###############################################################################
#                       Removing wedge foreground modes                       #
###############################################################################


def blackman_harris_taper(x: np.ndarray) -> np.ndarray:
    """
    Applies Blackman-Harris taper function to along line of sight (axis 0) of a
    3D lightcone.
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168

    bh = lambda n: a0 - \
                   a1 * np.cos(2*np.pi * n/x.shape[0]) + \
                   a2 * np.cos(4*np.pi * n/x.shape[0]) - \
                   a3 * np.cos(6*np.pi * n/x.shape[0])

    bh_window = np.array([bh(n) for n in range(x.shape[0])])

    return x * bh_window.reshape(-1, 1, 1)


def compute_wedge_boundary(z: float, 
                           fov_angle: float = 90.) -> float:
    """
    Computes the wedge boundary at a given redshift and fov angle.
    Formula given in Prelogovic et. al 2021 Eq (4)
    ----------
    Params:
    :z:         Redshift for wedge angle.
    :fov_angle: Angle of observation. Defaults to horizon limit.
    """

    E = lambda z: np.sqrt(O_m * np.power(1+z, 3) + O_d)
    E_inv = lambda z: 1/(np.sqrt(O_m * np.power(1+z, 3) + O_d))

    a = E(z)
    b, _ = quad(E_inv, 0, z)

    return (np.sin(fov_angle*np.pi/180) * a * b) / (1+z)


def sweep(x: np.ndarray, 
          z: float,
          fov_angle: float = 90.) -> np.ndarray:
    """
    Removes wedge region in cylindrical Fourier space.
    ----------
    Parameters
    :x:         Fourier-transformed lightcone
    :z:         Redshift for wedge-angle calculation
    :fov_angle: Horizon cut
    """

    assert x.shape[1] == x.shape[2], \
           f"expected square base but got shape {x.shape[1:]}"

    eps = 1e-18
    mid = x.shape[0]//2     # middle along los_direction
    core = x.shape[1]//2

    wedge_boundary_slope = compute_wedge_boundary(z, fov_angle)

    # Accounts for difference in resolution along LoS and transverse axes
    delta = x.shape[0] / x.shape[1]

    for i in range(mid):
        # Get threshold value for k_perp in Mpc
        k_parallel = i 
        radius = int(np.ceil(k_parallel / (wedge_boundary_slope + eps)))

        # Remove a box of width 2*radius
        x[mid+i, core+radius+1:, :] = 0j
        x[mid+i, :core-radius-2, :] = 0j
        x[mid+i, :, core+radius+1:] = 0j
        x[mid+i, :, :core-radius-2] = 0j
        x[mid-i-1, core+radius+1:, :] = 0j
        x[mid-i-1, :core-radius-2, :] = 0j
        x[mid-i-1, :, core+radius+1:] = 0j
        x[mid-i-1, :, :core-radius-2] = 0j

        for j in range(radius+2):
            for k in range(radius+2):
                k_perp = np.sqrt(j**2 + k**2)
                if (delta*k_perp) > radius:
                    x[mid+i, core+j, core+k] = 0j
                    x[mid-i-1, core+j, core+k] = 0j
                    x[mid+i, core-j-1, core+k] = 0j
                    x[mid-i-1, core-j-1, core+k] = 0j
                    x[mid+i, core+j, core-k-1] = 0j
                    x[mid-i-1, core+j, core-k-1] = 0j
                    x[mid+i, core-j-1, core-k-1] = 0j
                    x[mid-i-1, core-j-1, core-k-1] = 0j

    return x


###############################################################################
#                 Removing spectrally smooth foreground modes                 #
###############################################################################

def compute_smooth_kperp_cutoff(zmin: float, 
                                zmax: float) -> float:
    """
    Computes the maximum frequency to remove for spectrally smooth
    foregrounds, according to Equation (A10) of Liu et. al (2018).
    ----------
    Params:
    :zmin: minimum redshift of lightcone
    :zmax: maximum redshift of lightcone
    ----------
    Returns:
    :k_perp: cutoff index for maximum frequency to remove
    """

    E = lambda z: np.sqrt(O_m * np.power(1+z, 3) + O_d)
    nu_from = lambda z: NU_21 / (1+z)

    k_perp = (2*np.pi * H_0 * E(zmin) * NU_21) / \
             (LIGHTSPEED * np.power(1+zmin, 2) * (nu_from(zmin) - nu_from(zmax)))

    return k_perp


def bar(x: np.ndarray, 
        zmin: float,
        zmax: float,
        mpc_res) -> np.ndarray:
    """
    Removes smooth foreground contamination from 21cm lightcone.
    ----------
    Params:
    :x:       Fourier-transformed lightcone
    :zmin:    Minimum redshift of the lightcone
    :zmax:    Maximum redshift of the lightcone
    :mpc_res: Quotient of the size of axes 0 and 1
    ----------
    Returns:
    :x: Fourier-transformed lightcone with smooth foregrounds removed.
    """

    k_perp = compute_smooth_kperp_cutoff(zmin, zmax)
    mpc_per_px = 2 * np.pi / (x.shape[0] * mpc_res)

    maximum = int(np.round(k_perp / mpc_per_px))

    center = x.shape[0]//2
    x[center-maximum:center+maximum] = 0

    return x
