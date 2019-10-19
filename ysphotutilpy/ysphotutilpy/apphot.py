from warnings import warn

import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
from astropy.table import hstack
from photutils import aperture_photometry as apphot

from .background import sky_fit

__all__ = ["apphot_annulus"]

# TODO: Put centroiding into this apphot_annulus ?


def apphot_annulus(ccd, aperture, annulus, t_exposure=None,
                   exposure_key="EXPTIME", error=None, mask=None,
                   sky_keys={}, t_exposure_unit=u.s, verbose=False,
                   **kwargs):
    ''' Do aperture photometry using annulus.
    Parameters
    ----------
    ccd: CCDData
        The data to be photometried. Preferably in ADU.
    aperture, annulus: photutils aperture and annulus object
        The aperture and annulus to be used for aperture photometry.
    exposure_key: str
        The key for exposure time. Together with ``t_exposure_unit``, the
        function will normalize the signal to exposure time. If ``t_exposure``
        is not None, this will be ignored.
    error: array-like or Quantity, optional
        See ``photutils.aperture_photometry`` documentation.
        The pixel-wise error map to be propagated to magnitued error.
    sky_keys: dict
        kwargs of ``sky_fit``. Mostly one doesn't change the default setting,
        so I intentionally made it to be dict rather than usual kwargs, etc.
    **kwargs:
        kwargs for ``photutils.aperture_photometry``.

    Returns
    -------
    phot_f: astropy.table.Table
        The photometry result.
    '''
    _ccd = ccd.copy()

    if t_exposure is None:
        t_exposure = ccd.header[exposure_key]

    if error is not None:
        if verbose:
            print("Ignore any uncertainty extension in the original CCD "
                  + "and use provided error.")
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data

    else:
        try:
            err = ccd.uncertainty.array
        except AttributeError:
            if verbose:
                warn("Couldn't find Uncertainty extension in ccd. "
                     + "Will not calculate errors.")
            err = np.zeros_like(_ccd.data)

    if mask is not None:
        if _ccd.mask is not None:
            if verbose:
                warn("ccd contains mask, so given mask will be added to it.")
            _ccd.mask += mask
        else:
            _ccd.mask = mask

    skys = sky_fit(_ccd, annulus, **sky_keys)
    try:
        n_ap = aperture.area()
    except TypeError:  # as of photutils 0.7
        n_ap = aperture.area
    phot = apphot(_ccd.data, aperture, mask=_ccd.mask, error=err, **kwargs)
    # If we use ``_ccd``, photutils deal with the unit, and the lines below
    # will give a lot of headache for units. It's not easy since aperture
    # can be pixel units or angular units (Sky apertures).
    # ysBach 2018-07-26

    phot_f = hstack([phot, skys])

    phot_f["source_sum"] = phot_f["aperture_sum"] - n_ap * phot_f["msky"]

    # see, e.g., http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?radprof.hlp
    # Poisson + RDnoise + dark + digitization noise:
    var_errmap = phot_f["aperture_sum_err"]**2
    # Sum of n_ap Gaussians (kind of random walk):
    var_skyrand = n_ap * phot_f["ssky"]**2
    # "systematic" uncertainty in the msky value:
    var_skysyst = (n_ap * phot_f['ssky'])**2 / phot_f['nsky']

    phot_f["source_sum_err"] = np.sqrt(var_errmap + var_skyrand + var_skysyst)

    phot_f["mag"] = -2.5 * np.log10(phot_f['source_sum'] / t_exposure)
    phot_f["merr"] = (2.5 / np.log(10)
                      * phot_f["source_sum_err"] / phot_f['source_sum'])

    return phot_f
