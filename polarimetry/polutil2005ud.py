from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

PI = np.pi
D2R = PI / 180

DATAPATH = Path('data')
SAVEPATH = Path('figs')
SAVEPATH.mkdir(exist_ok=True)

dats = pd.read_csv(DATAPATH/"pol_ud_data.csv", sep=',')
dats.insert(loc=7, column="dPr", value=dats["dP"])
# dats["dPr"] = np.max(np.array([dats["Pr"]*0.05, [0.1]*len(dats), dats["dPr"]]).T, axis=1)
dats = dats[dats["dPr"] < 10]
dats = dats.sort_values("alpha")
dats = dats.reset_index(drop=True)

alpha = dats["alpha"].to_numpy()
polr = dats["Pr"].to_numpy()
dpolr = dats["dPr"].to_numpy()

dats_msi = dats[dats["obs"] == 'MSI']
dats_oth = dats[dats["obs"] != 'MSI']
alpha_msi = dats_msi["alpha"].to_numpy()
alpha_oth = dats_oth["alpha"].to_numpy()
polr_msi = dats_msi["Pr"].to_numpy()
polr_oth = dats_oth["Pr"].to_numpy()
dpolr_msi = dats_msi["dPr"].to_numpy()
dpolr_oth = dats_oth["dPr"].to_numpy()


class Param:
    def __init__(self, name, low, upp, p0):
        self.name = str(name)
        self.low = low
        self.upp = upp
        self.p0 = p0

    def __str__(self):
        return "Parameter {} in [{}, {}]; default {}".format(self.name, self.low, self.upp, self.p0)


pars_trigp_b = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    c1=Param('c1', 1.e-6, 1.e+1, 1),
    c2=Param('c2', 1.e-6, 1.e+1, 1)
)

pars_trigp_f = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    c1=Param('c1', -1.e+1, 1.e+1, 1),
    c2=Param('c2', -1.e+1, 1.e+1, 1)
)

pars_shesp = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    k1=Param('k1', 1.e-8, 1., 0.001),
    k2=Param('k2', 1.e-8, 1., 1.e-5),
    k0=Param('k0', 1.e-8, 1., 1.e-5)
)

pars_appsp = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    k1=Param('k1', 1.e-8, 1., 0.001)
)

pars_sgbip = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    kn=Param('kn', -1., 1., 0.001),
    kp=Param('kp', 1.e-8, 1., 0.001)
)

p0_trigp_b = tuple([p.p0 for p in pars_trigp_b.values()])
p0_trigp_f = tuple([p.p0 for p in pars_trigp_f.values()])
p0_shesp = tuple([p.p0 for p in pars_shesp.values()])
p0_appsp = tuple([p.p0 for p in pars_appsp.values()])
p0_sgbip = tuple([p.p0 for p in pars_sgbip.values()])

bounds_trigp_b = (tuple([p.low for p in pars_trigp_b.values()]),
                  tuple([p.upp for p in pars_trigp_b.values()]))
bounds_trigp_f = (tuple([p.low for p in pars_trigp_f.values()]),
                  tuple([p.upp for p in pars_trigp_f.values()]))
bounds_shesp = (tuple([p.low for p in pars_shesp.values()]),
                tuple([p.upp for p in pars_shesp.values()]))
bounds_appsp = (tuple([p.low for p in pars_appsp.values()]),
                tuple([p.upp for p in pars_appsp.values()]))
bounds_sgbip = (tuple([p.low for p in pars_sgbip.values()]),
                tuple([p.upp for p in pars_sgbip.values()]))

pars = dict(trigp_b=pars_trigp_b, trigp_f=pars_trigp_f, shesp=pars_shesp, appsp=pars_appsp, sgbip=pars_sgbip)
p0 = dict(trigp_b=p0_trigp_b, trigp_f=p0_trigp_f, shesp=p0_shesp, appsp=p0_appsp, sgbip=p0_sgbip)
bounds = dict(trigp_b=bounds_trigp_b, trigp_f=bounds_trigp_f, shesp=bounds_shesp, appsp=bounds_appsp, sgbip=bounds_sgbip)


@nb.njit  # (parallel=True)
def cos_deg(x):
    return np.cos(x * D2R)


@nb.njit  # (parallel=True)
def sin_deg(x):
    return np.sin(x * D2R)


def trigp(x, h=p0['trigp_b'][0], a0=p0['trigp_b'][1], c1=p0['trigp_b'][2], c2=p0['trigp_b'][3]):
    ''' Lumme-Muinonen function in pure python mode.
    '''
    term1 = (sin_deg(x) / sin_deg(a0))**c1
    term2 = (cos_deg(x / 2) / cos_deg(a0 / 2))**c2
    term3 = sin_deg(x - a0)
    Pr = h / D2R * term1 * term2 * term3
    return Pr


@nb.njit  # (parallel=True)
def _nb_trigp(x, h=p0['trigp_b'][0], a0=p0['trigp_b'][1], c1=p0['trigp_b'][2], c2=p0['trigp_b'][3]):
    ''' Lumme-Muinonen function in numba mode.
    '''
    term1 = (sin_deg(x) / sin_deg(a0))**c1
    term2 = (cos_deg(x / 2) / cos_deg(a0 / 2))**c2
    term3 = sin_deg(x - a0)
    Pr = h / D2R * term1 * term2 * term3
    return Pr


@nb.njit(parallel=True)
def do_trigp(x, y, yerr, arr_h, arr_a0, arr_c1, arr_c2, arr_chi2, arr_amax,
             arr_Pmax, arr_amin, arr_Pmin):
    ''' To calculate everything needed for the analysis.
    Parameters
    ----------
    x, y, yerr : array-like
        The phase angle [˚], Pr [%], and dPr [%].
    arr_h, arr_c1, arr_c2, arr_a0 : array-like
        The ``trace``d arrays of the four parameters of Lumme-Muinonen function from pymc3.
    arr_chi2, arr_amax, arr_Pmax, arr_amin, arr_Pmin : array-like
        The empty arrays of chi-square, min/max phase angle (``a``) and the polarization degree
        (``P``). Must have the same length as the arrays given above. (It will not give error if these
        are longer than the above ones, but...)
    '''
    xx_min = np.arange(2, 15, 0.01)
    xx_max = np.arange(80, 140, 0.01)

    for i in nb.prange(arr_h.shape[0]):
        h = arr_h[i]
        a0 = arr_a0[i]
        c1 = arr_c1[i]
        c2 = arr_c2[i]
        resid = y - _nb_trigp(x, h=h, a0=a0, c1=c1, c2=c2)
        chi2 = np.sum((resid / yerr)**2)
        amax, Pmax = trigp_max(xx_max, h=h, a0=a0, c1=c1, c2=c2)
        amin, Pmin = trigp_min(xx_min, h=h, a0=a0, c1=c1, c2=c2)

        arr_chi2[i] = chi2
        arr_amax[i] = amax
        arr_Pmax[i] = Pmax
        arr_amin[i] = amin
        arr_Pmin[i] = Pmin


@nb.njit  # (parallel=True)
def trigp_min(xx, h=p0['trigp_b'][0], a0=p0['trigp_b'][1], c1=p0['trigp_b'][2], c2=p0['trigp_b'][3]):
    """ Calculates the minimum phase angle/P degree from given parameters.
    """
    minimum = 1
    for i in range(xx.shape[0]):
        p = _nb_trigp(xx[i], h, a0, c1, c2)
        if p < minimum:
            minimum = p
        else:
            break
    return (xx[i - 1], minimum)


@nb.njit  # (parallel=True)
def trigp_max(xx, h=p0['trigp_b'][0], a0=p0['trigp_b'][1], c1=p0['trigp_b'][2], c2=p0['trigp_b'][3]):
    """ Calculates the maximum phase angle/P degree from given parameters.
    """
    maximum = -1
    for i in range(xx.shape[0]):
        p = _nb_trigp(xx[i], h, a0, c1, c2)
        if p > maximum:
            maximum = p
        else:
            break
    return (xx[i - 1], maximum)


# ********************************************************************************************************** #
# *                                      SHESTOPALOV FUNCTION -- FULL                                      * #
# ********************************************************************************************************** #

def shesp(x, h=p0['shesp'][0], a0=p0['shesp'][1], k1=p0['shesp'][2], k2=p0['shesp'][3],
          k0=p0['shesp'][4]):
    ''' Shestopalov function in pure python mode.
    '''
    term1 = (1 - np.exp(-k1*x))/(1 - np.exp(-k1*a0))
    term2 = (1 - np.exp(-k0*(x - a0)))/k0
    term3 = (1 - np.exp(-k2*(x - 180)))/(1 - np.exp(-k2*(a0 - 180)))
    # No D2R needed for h because term1*term2*term3 has unit of [deg] so h is already %/deg.
    Pr = h*term1*term2*term3
    return Pr


@nb.njit  # (parallel=True)
def _nb_shesp(x, h=p0['shesp'][0], a0=p0['shesp'][1], k1=p0['shesp'][2], k2=p0['shesp'][3], k0=p0['shesp'][4]):
    ''' Shestopalov function in numba mode.
    '''
    term1 = (1 - np.exp(-k1*x))/(1 - np.exp(-k1*a0))
    term2 = (1 - np.exp(-k0*(x - a0)))/k0
    term3 = (1 - np.exp(-k2*(x - 180)))/(1 - np.exp(-k2*(a0 - 180)))
    # No D2R needed for h because term1*term2*term3 has unit of [deg] so h is already %/deg.
    Pr = h*term1*term2*term3
    return Pr


@nb.njit(parallel=True)
def do_shesp(x, y, yerr, arr_h, arr_a0, arr_k1, arr_k2, arr_k0, arr_chi2, arr_amax,
             arr_Pmax, arr_amin, arr_Pmin):
    ''' To calculate everything needed for the analysis.
    Parameters
    ----------
    x, y, yerr : array-like
        The phase angle [˚], Pr [%], and dPr [%].
    arr_h, arr_a0, arr_k0, arr_k1, arr_k2 : array-like
        The ``trace``d arrays of the four parameters of Shestopalov function from pymc3.
    arr_chi2, arr_amax, arr_Pmax, arr_amin, arr_Pmin : array-like
        The empty arrays of chi-square, min/max phase angle (``a``) and the polarization degree
        (``P``). Must have the same length as the arrays given above. (It will not give error if these
        are longer than the above ones, but...)
    '''
    xx_min = np.arange(2, 15, 0.01)
    xx_max = np.arange(80, 140, 0.01)

    for i in nb.prange(arr_h.shape[0]):
        h = arr_h[i]
        a0 = arr_a0[i]
        k1 = arr_k1[i]
        k2 = arr_k2[i]
        k0 = arr_k0[i]
        resid = y - _nb_shesp(x, h=h, a0=a0, k1=k1, k2=k2, k0=k0)
        chi2 = np.sum((resid / yerr)**2)
        amax, Pmax = shesp_max(xx_max, h=h, a0=a0, k1=k1, k2=k2, k0=k0)
        amin, Pmin = shesp_min(xx_min, h=h, a0=a0, k1=k1, k2=k2, k0=k0)

        arr_chi2[i] = chi2
        arr_amax[i] = amax
        arr_Pmax[i] = Pmax
        arr_amin[i] = amin
        arr_Pmin[i] = Pmin


@nb.njit  # (parallel=True)
def shesp_min(xx, h=p0['shesp'][0], a0=p0['shesp'][1], k1=p0['shesp'][2], k2=p0['shesp'][3], k0=p0['shesp'][4]):
    """ Calculates the minimum phase angle/P degree from given parameters.
    """
    minimum = 1
    for i in range(xx.shape[0]):
        p = _nb_shesp(xx[i], h, a0, k1, k2, k0)
        if p < minimum:
            minimum = p
        else:
            break
    return (xx[i - 1], minimum)


@nb.njit  # (parallel=True)
def shesp_max(xx, h=p0['shesp'][0], a0=p0['shesp'][1], k1=p0['shesp'][2], k2=p0['shesp'][3], k0=p0['shesp'][4]):
    """ Calculates the maximum phase angle/P degree from given parameters.
    """
    maximum = -1
    for i in range(xx.shape[0]):
        p = _nb_shesp(xx[i], h, a0, k1, k2, k0)
        if p > maximum:
            maximum = p
        else:
            break
    return (xx[i - 1], maximum)


# ********************************************************************************************************** #
# *                                   SHESTOPALOV FUNCTION -- 3-PARAMETER                                  * #
# ********************************************************************************************************** #


def appsp(x, h=p0['appsp'][0], a0=p0['appsp'][1], k1=p0['appsp'][2]):
    ''' Approximated Shestopalov function in pure python mode.
    '''
    term1 = (1 - np.exp(-k1*x))/(1 - np.exp(-k1*a0))
    term2 = (x - a0)
    term3 = (x - 180)/(a0 - 180)
    # No D2R needed for h because term1*term2*term3 has unit of [deg] so h is already %/deg.
    Pr = h*term1*term2*term3
    return Pr


@nb.njit  # (parallel=True)
def _nb_appsp(x, h=p0['appsp'][0], a0=p0['appsp'][1], k1=p0['appsp'][2]):
    ''' Approximated Shestopalov function in numba mode.
    '''
    term1 = (1 - np.exp(-k1*x))/(1 - np.exp(-k1*a0))
    term2 = (x - a0)
    term3 = (x - 180)/(a0 - 180)
    # No D2R needed for h because term1*term2*term3 has unit of [deg] so h is already %/deg.
    Pr = h*term1*term2*term3
    return Pr


@nb.njit(parallel=True)
def do_appsp(x, y, yerr, arr_h, arr_a0, arr_k1, arr_chi2, arr_amax,
             arr_Pmax, arr_amin, arr_Pmin):
    ''' To calculate everything needed for the analysis.
    Parameters
    ----------
    x, y, yerr : array-like
        The phase angle [˚], Pr [%], and dPr [%].
    arr_h, arr_a0, arr_k1 : array-like
        The ``trace``d arrays of the four parameters of Shestopalov function from pymc3.
    arr_chi2, arr_amax, arr_Pmax, arr_amin, arr_Pmin : array-like
        The empty arrays of chi-square, min/max phase angle (``a``) and the polarization degree
        (``P``). Must have the same length as the arrays given above. (It will not give error if these
        are longer than the above ones, but...)
    '''
    xx_min = np.arange(2, 15, 0.01)
    xx_max = np.arange(80, 140, 0.01)

    for i in nb.prange(arr_h.shape[0]):
        h = arr_h[i]
        a0 = arr_a0[i]
        k1 = arr_k1[i]
        resid = y - _nb_appsp(x, h=h, a0=a0, k1=k1)
        chi2 = np.sum((resid / yerr)**2)
        amax, Pmax = appsp_max(xx_max, h=h, a0=a0, k1=k1)
        amin, Pmin = appsp_min(xx_min, h=h, a0=a0, k1=k1)

        arr_chi2[i] = chi2
        arr_amax[i] = amax
        arr_Pmax[i] = Pmax
        arr_amin[i] = amin
        arr_Pmin[i] = Pmin


@nb.njit  # (parallel=True)
def appsp_min(xx, h=p0['appsp'][0], a0=p0['appsp'][1], k1=p0['appsp'][2]):
    """ Calculates the minimum phase angle/P degree from given parameters.
    """
    minimum = 1
    for i in range(xx.shape[0]):
        p = _nb_appsp(xx[i], h, a0, k1)
        if p < minimum:
            minimum = p
        else:
            break
    return (xx[i - 1], minimum)


@nb.njit  # (parallel=True)
def appsp_max(xx, h=p0['appsp'][0], a0=p0['appsp'][1], k1=p0['appsp'][2]):
    """ Calculates the maximum phase angle/P degree from given parameters.
    """
    maximum = -1
    for i in range(xx.shape[0]):
        p = _nb_appsp(xx[i], h, a0, k1)
        if p > maximum:
            maximum = p
        else:
            break
    return (xx[i - 1], maximum)


# ********************************************************************************************************** #
# *               SHESTOPALOV-GOLUBEVA-BACH-ISHIGURO FUNCTION -- 4-param                                   * #
# ********************************************************************************************************** #

def sgbip(x, h=p0['sgbip'][0], a0=p0['sgbip'][1], kn=p0['sgbip'][2], kp=p0['sgbip'][3]):
    ''' Shestopalov function in pure python mode.
    '''
    term0 = h*(x - a0)*(x - 180)/(a0 - 180)
    term1 = (1 - np.exp(-kn*x))/(1 - np.exp(-kn*a0))
    term2 = (1 + np.exp(-kp*x))/(1 + np.exp(-kp*a0))
    # No D2R needed for h because term1*term2*term3 has unit of [deg] so h is already %/deg.
    Pr = term0*term1*term2
    return Pr


@nb.njit  # (parallel=True)
def _nb_sgbip(x, h=p0['sgbip'][0], a0=p0['sgbip'][1], kn=p0['sgbip'][2], kp=p0['sgbip'][3]):
    ''' Shestopalov function in numba mode.
    '''
    term0 = h*(x - a0)*(x - 180)/(a0 - 180)
    term1 = (1 - np.exp(-kn*x))/(1 - np.exp(-kn*a0))
    term2 = (1 + np.exp(-kp*x))/(1 + np.exp(-kp*a0))
    # No D2R needed for h because term1*term2*term3 has unit of [deg] so h is already %/deg.
    Pr = term0*term1*term2
    return Pr


@nb.njit(parallel=True)
def do_sgbip(x, y, yerr, arr_h, arr_a0, arr_kn, arr_kp, arr_chi2, arr_amax,
             arr_Pmax, arr_amin, arr_Pmin):
    ''' To calculate everything needed for the analysis.
    Parameters
    ----------
    x, y, yerr : array-like
        The phase angle [˚], Pr [%], and dPr [%].
    arr_h, arr_a0, arr_kn, arr_kp : array-like
        The ``trace``d arrays of the four parameters of Shestopalov function from pymc3.
    arr_chi2, arr_amax, arr_Pmax, arr_amin, arr_Pmin : array-like
        The empty arrays of chi-square, min/max phase angle (``a``) and the polarization degree
        (``P``). Must have the same length as the arrays given above. (It will not give error if these
        are longer than the above ones, but...)
    '''
    xx_min = np.arange(2, 15, 0.01)
    xx_max = np.arange(80, 140, 0.01)

    for i in nb.prange(arr_h.shape[0]):
        h = arr_h[i]
        a0 = arr_a0[i]
        kn = arr_kn[i]
        kp = arr_kp[i]
        resid = y - _nb_sgbip(x, h=h, a0=a0, kn=kn, kp=kp)
        chi2 = np.sum((resid / yerr)**2)
        amax, Pmax = sgbip_max(xx_max, h=h, a0=a0, kn=kn, kp=kp)
        amin, Pmin = sgbip_min(xx_min, h=h, a0=a0, kn=kn, kp=kp)

        arr_chi2[i] = chi2
        arr_amax[i] = amax
        arr_Pmax[i] = Pmax
        arr_amin[i] = amin
        arr_Pmin[i] = Pmin


@nb.njit  # (parallel=True)
def sgbip_min(xx, h=p0['sgbip'][0], a0=p0['sgbip'][1], kn=p0['sgbip'][2], kp=p0['sgbip'][3]):
    """ Calculates the minimum phase angle/P degree from given parameters.
    """
    minimum = 1
    for i in range(xx.shape[0]):
        p = _nb_sgbip(xx[i], h, a0, kn, kp)
        if p < minimum:
            minimum = p
        else:
            break
    return (xx[i - 1], minimum)


@nb.njit  # (parallel=True)
def sgbip_max(xx, h=p0['sgbip'][0], a0=p0['sgbip'][1], kn=p0['sgbip'][2], kp=p0['sgbip'][3]):
    """ Calculates the maximum phase angle/P degree from given parameters.
    """
    maximum = -1
    for i in range(xx.shape[0]):
        p = _nb_sgbip(xx[i], h, a0, kn, kp)
        if p > maximum:
            maximum = p
        else:
            break
    return (xx[i - 1], maximum)


