"""
A collection of temporary utilities, and likely be removed if similar
functionality can be achieved by pre-existing packages.
"""
import numpy as np


__all__ = ["Gaussian2D_correct"]


def normalize(num, lower=0, upper=360, b=False):
    """Normalize number to range [lower, upper) or [lower, upper].
    From phn: https://github.com/phn/angles
    Parameters
    ----------
    num : float
        The number to be normalized.
    lower : int
        Lower limit of range. Default is 0.
    upper : int
        Upper limit of range. Default is 360.
    b : bool
        Type of normalization. Default is False. See notes.
        When b=True, the range must be symmetric about 0.
        When b=False, the range must be symmetric about 0 or ``lower`` must
        be equal to 0.
    Returns
    -------
    n : float
        A number in the range [lower, upper) or [lower, upper].
    Raises
    ------
    ValueError
      If lower >= upper.
    Notes
    -----
    If the keyword `b == False`, then the normalization is done in the
    following way. Consider the numbers to be arranged in a circle,
    with the lower and upper ends sitting on top of each other. Moving
    past one limit, takes the number into the beginning of the other
    end. For example, if range is [0 - 360), then 361 becomes 1 and 360
    becomes 0. Negative numbers move from higher to lower numbers. So,
    -1 normalized to [0 - 360) becomes 359.
    When b=False range must be symmetric about 0 or lower=0.
    If the keyword `b == True`, then the given number is considered to
    "bounce" between the two limits. So, -91 normalized to [-90, 90],
    becomes -89, instead of 89. In this case the range is [lower,
    upper]. This code is based on the function `fmt_delta` of `TPM`.
    When b=True range must be symmetric about 0.
    Examples
    --------
    >>> normalize(-270,-180,180)
    90.0
    >>> import math
    >>> math.degrees(normalize(-2*math.pi,-math.pi,math.pi))
    0.0
    >>> normalize(-180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180, b=True)
    180.0
    >>> normalize(181,-180,180)
    -179.0
    >>> normalize(181, -180, 180, b=True)
    179.0
    >>> normalize(-180,0,360)
    180.0
    >>> normalize(36,0,24)
    12.0
    >>> normalize(368.5,-180,180)
    8.5
    >>> normalize(-100, -90, 90)
    80.0
    >>> normalize(-100, -90, 90, b=True)
    -80.0
    >>> normalize(100, -90, 90, b=True)
    80.0
    >>> normalize(181, -90, 90, b=True)
    -1.0
    >>> normalize(270, -90, 90, b=True)
    -90.0
    >>> normalize(271, -90, 90, b=True)
    -89.0
    """
    if lower >= upper:
        ValueError("lower must be lesser than upper")
    if not b:
        if not ((lower + upper == 0) or (lower == 0)):
            raise ValueError(
                'When b=False lower=0 or range must be symmetric about 0.')
    else:
        if not (lower + upper == 0):
            raise ValueError('When b=True range must be symmetric about 0.')

    from math import floor, ceil
    # abs(num + upper) and abs(num - lower) are needed, instead of
    # abs(num), since the lower and upper limits need not be 0. We need
    # to add half size of the range, so that the final result is lower +
    # <value> or upper - <value>, respectively.
    res = num
    if not b:
        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if num == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num

    res *= 1.0  # Make all numbers float, to be consistent

    return res


def Gaussian2D_correct(model, theta_lower=-np.pi/2, theta_upper=np.pi/2):
    ''' Sets x = semimajor axis and theta to be in [-pi/2, pi/2] range.
    Example
    -------
    >>> from astropy.modeling.functional_models import Gaussian2D
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from yspy.util import astropy_util as au
    >>> gridsize = np.zeros((40, 60))
    >>> common = dict(x_mean=20, y_mean=20, x_stddev=5)
    >>> y, x = np.mgrid[:gridsize.shape[0], :gridsize.shape[1]]
    >>> theta_arr = [-12345.678, -100, -1, -0.1, 0, 0.1, 1, 100, 12345.678]
    >>> for sig_y in [-1, -0.1, 0.1, 1, 10]:
    >>>     for theta in theta_arr:
    >>>         g = Gaussian2D(**common, theta=theta)
    >>>         g_c = Gaussian2D_correct(g)
    >>>         f, ax = plt.subplots(3)
    >>>         ax[0].imshow(g(x, y), vmax=1, vmin=1.e-12)
    >>>         ax[1].imshow(g_c(x, y), vmax=1, vmin=1.e-12)
    >>>         ax[2].imshow(g(x, y) - g_c(x, y), vmin=1.e-20, vmax=1.e-12)
    >>>         np.testing.assert_almost_equal(g(x, y) - g_c(x, y), gridsize)
    >>>         plt.pause(0.1)
    You may see some patterns in the residual image, they are < 10**(-13).
    '''
    # I didn't use ``Gaussian2D`` directly, becaus GaussianConst2D from
    # photutils may also be used.
    new_model = model.__class__(*model.parameters)
    sig_x = np.abs(model.x_stddev.value)
    sig_y = np.abs(model.y_stddev.value)
    theta = model.theta.value

    if sig_x > sig_y:
        theta_norm = normalize(theta, theta_lower, theta_upper)
        new_model.x_stddev.value = sig_x
        new_model.y_stddev.value = sig_y
        new_model.theta.value = theta_norm

    else:
        theta_norm = normalize(theta + np.pi/2, theta_lower, theta_upper)
        new_model.x_stddev.value = sig_y
        new_model.y_stddev.value = sig_x
        new_model.theta.value = theta_norm

    return new_model


'''
from warnings import warn
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import CONSTRAINTS_DOC, Const2D, Moffat2D
from astropy.utils.exceptions import AstropyUserWarning

# TODO: Elliptical moffat...
# https://iraf.net/irafhelp.php?val=daopars&help=Help+Page
class MoffatConst2D(Fittable2DModel):
    """
    A model for a 2D Moffat plus a constant.

    Parameters
    ----------
    constant : float
        Value of the constant.
    amplitude : float
        Amplitude of the model.
    x_0 : float
        x position of the maximum of the Moffat model.
    y_0 : float
        y position of the maximum of the Moffat model.
    gamma : float
        Core width of the Moffat model.
    alpha : float
        Power index of the Moffat model.
    """

    constant = Parameter(default=1)
    amplitude = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    gamma = Parameter(default=1)
    alpha = Parameter(default=1)

    @staticmethod
    def evaluate(x, y, constant, amplitude, x_0, y_0, gamma, alpha):
        """Two dimensional Gaussian plus constant function."""

        model = Const2D(constant)(x, y) + Moffat2D(amplitude, x_0, y_0,
                                                   gamma, alpha)(x, y)
        return model


MoffatConst2D.__doc__ += CONSTRAINTS_DOC


def fit_2dmoffat(data, error=None, mask=None):
    """
    Fit a 2D Moffat plus a constant to a 2D image.

    Invalid values (e.g. NaNs or infs) in the ``data`` or ``error``
    arrays are automatically masked.  The mask for invalid values
    represents the combination of the invalid-value masks for the
    ``data`` and ``error`` arrays.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    result : A `MoffatConst2D` model instance.
        The best-fitting Moffat 2D model.
    """

    from ..morphology import data_properties  # prevent circular imports

    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warn('Input data contains input values (e.g. NaNs or infs), '
             'which were automatically masked.', AstropyUserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
        weights = 1.0 / error.clip(min=1.e-30)
    else:
        weights = np.ones(data.shape)

    if np.ma.count(data) < 7:
        raise ValueError('Input data must have a least 7 unmasked values to '
                         'fit a 2D Moffat plus a constant.')

    # assign zero weight to masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 0.

    mask = data.mask
    data.fill_value = 0.0
    data = data.filled()

    # Subtract the minimum of the data as a crude background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties (moments from negative data
    # values can yield undefined Gaussian parameters, e.g. x/y_stddev).
    props = data_properties(data - np.min(data), mask=mask)

    init_const = 0.  # subtracted data minimum above
    init_amplitude = np.ptp(data)
    g_init = GaussianConst2D(constant=init_const, amplitude=init_amplitude,
                             x_mean=props.xcentroid.value,
                             y_mean=props.ycentroid.value,
                             x_stddev=props.semimajor_axis_sigma.value,
                             y_stddev=props.semiminor_axis_sigma.value,
                             theta=props.orientation.value)
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)

    return gfit
'''
