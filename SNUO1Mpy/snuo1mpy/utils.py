# import warnings
from pathlib import Path
import shutil
import os

from astropy.io.fits import Card

__all__ = ["MEDCOMB_KEYS", "SITE_HORIZONS", "GAIN_EPADU", "RDNOISE_E",
           "KEYMAP", "USEFUL_KEYS",
           "cards_gain_rdnoise"]

MEDCOMB_KEYS = dict(overwrite=True,
                    unit=None,
                    combine_method="median",
                    reject_method=None,
                    combine_uncertainty_function=None)

# The ``location``` for astroquery.jplhorizons
# https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html
#   longitude in degrees (East positive, West negative)
#   latitude in degrees (North positive, South negative)
#   elevation in km above the reference ellipsoid
SITE_HORIZONS = dict(lon=126.95333, lat=37.45694, elevation=0.2)

# Gain and rdnoise from SAO1-m package (@lim9gu)
# https://github.com/lim9gu/SAO1-m/commit/bd8a931265bd611fb1f4152fdd802792aa702cce
# Retrieved in 2019 May.
GAIN_EPADU = {"STX16803": 1.3600000143051147,
              "Kepler": 1.5}

RDNOISE_E = {"STX16803": 9.0,
             "Kepler": None}

# <FITS Standard> : <our observatory>
KEYMAP = {"EXPTIME": 'EXPTIME',
          "GAIN": 'EGAIN',
          "OBJECT": 'OBJECT',
          "FILTER": 'FILTER',
          "EQUINOX": None,
          "DATE-OBS": 'DATE-OBS',
          "RDNOISE": None}


USEFUL_KEYS = ["DATE-OBS", "FILTER", "OBJECT", "EXPTIME", "IMAGETYP",
               "AIRMASS", "XBINNING", "YBINNING", "CCD-TEMP", "SET-TEMP",
               "OBJCTRA", "OBJCTDEC", "OBJCTALT"]


def reset_dir(topdir):
    topdir = Path(topdir)
    dirsattop = list(topdir.iterdir())
    dirsatraw = list((topdir / "rawdata").iterdir())

    for path in dirsattop:
        if path.name != "rawdata":
            if path.is_dir() and not path.name.startswith("."):
                shutil.rmtree(path)
            else:
                os.remove(path)

    for path in dirsatraw:
        if ((path.is_dir()) and (path.name != "archive")
                and (path.name != "useless") and not (path.name.startswith("."))):
            shutil.rmtree(path)
        else:
            fpaths = path.glob("*")
            for fpath in fpaths:
                os.rename(fpath, topdir / "rawdata" / fpath.name)

    shutil.rmtree(topdir / "rawdata" / "archive")
    shutil.rmtree(topdir / "rawdata" / "useless")


def cards_gain_rdnoise(instrument="STX16803", gain=None, rdnoise=None):
    ''' Returns Card list of gain and rdnoise
    Parameters
    ----------
    instrument : str, optional
        The instrument code. Currently one of ["STX16803", "FLI Kepler"].

    gain, rdnoise : float, optional
        The gain and read noise if you want to specify. Must be in the unit
        of electrons per ADU and electrons, respectively.
    '''
    cs = []

    if gain is None:
        gainstr = f"GAIN from the intrument {instrument}."
        gain = GAIN_EPADU[instrument]

    else:
        gainstr = f"GAIN provided by the user as {gain}."

    if rdnoise is None:
        rdnoisestr = f"RDNOISE from the intrument {instrument}."
        rdnoise = RDNOISE_E[instrument]
    else:
        rdnoisestr = f"RDNOISE rovided by the user as {rdnoise}."

    cs = [Card("GAIN", gain, "[e-/ADU] The electron gain factor."),
          Card("RDNOISE", rdnoise, "[e-] The (Gaussian) read noise."),
          Card("COMMENT", gainstr),
          Card("COMMENT", rdnoisestr)]

    return cs
