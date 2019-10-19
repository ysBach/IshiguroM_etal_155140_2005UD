import numpy as np

__all__ = ["reduced_mag"]


def reduced_mag(mag, r_hel, r_obs):
    return mag - 5 * np.log10(r_hel * r_obs)