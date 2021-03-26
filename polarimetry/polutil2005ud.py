from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.optimize import curve_fit

import ysvisutilpy2005ud as yvu

PI = np.pi
D2R = PI / 180

DATAPATH = Path('data')
SAVEPATH = Path('figs')
SAVEPATH.mkdir(exist_ok=True)

# ********************************************************************************************************** #
# *                                      COMBINE MSI AND DEVOGELE DATA                                     * #
# ********************************************************************************************************** #
dats = pd.read_csv(DATAPATH/"pol_ud_data.csv", sep=',')
dats.insert(loc=7, column="dPr", value=dats["dP"])

dat2 = pd.read_csv(DATAPATH/"2020PSJ.....1...15D.csv")
dat2 = dat2.loc[dat2["Facility"] == "Rozhen"]
# Aggregate data into similar-phase-angle cases, and only save weighted averages of such bins.
dates = Time(dat2["JD"], format="jd")
ndat2 = dict(date=[], alpha=[], Pr=[], dPr=[])
for center in [17, 24, 32]:
    near_center = np.abs(dat2["PA"] - center) < 2
    # print(dates[near_center].isot)
    _dat = dat2[near_center]
    _p = _dat["Pr"]
    _dp = _dat["err_Pr"]
    _isse = np.sum(1/_dp**2)  # Inverse Square Sum of Errors
    ndat2["date"].append(Time(np.median(dates[near_center].jd), format='jd').strftime("%Y-%m-%d"))
    ndat2["alpha"].append(np.median(_dat["PA"]))
    ndat2["Pr"].append(np.sum(_p/_dp**2)/_isse)
    ndat2["dPr"].append(1/np.sqrt(_isse))

ndat2 = pd.DataFrame.from_dict(ndat2)
ndat2["obs"] = "FoReRo2"

# Merge two
dats = pd.concat([dats, ndat2])

# dats["dPr"] = np.max(np.array([dats["Pr"]*0.05, [0.1]*len(dats), dats["dPr"]]).T, axis=1)
dats = dats[dats["dPr"] < 10]

dats = dats.sort_values("alpha")
dats = dats.reset_index(drop=True)

# ---------------------------------------------------------------------------------------------------------- #

# ********************************************************************************************************** #
# *                         MAKE THE DATAFRAME FOR OTHER ASTEROIDS (BIN WITH ALPHA)                        * #
# ********************************************************************************************************** #

_dat_ast = pd.read_csv("data/pol_data_other_asteroids.csv")
_dat_ast = _dat_ast.loc[(_dat_ast["filter"].isin(["0.65", "0.68"]))
                        | (_dat_ast["filter"].str.startswith("R"))]
_dats_ast = dict(label=[], midjd=[], alpha=[], Pr=[], dPr=[], filter=[], reference=[])

_dat_g = _dat_ast.groupby(["label", "reference", "filter"])
for (label, reference, filt), df in _dat_g:
    datetime = Time((df["date"].astype(str) + "T" + df["middletime"].astype(str)).tolist(), format="isot")
    df["jd"] = datetime.jd
    dt = np.ediff1d(datetime.jd)
    idxs = np.where(dt > 0.1)[0]  # indices where more then sudden time gap of 0.1+ days
    if len(idxs) > 0:
        idxs = [0] + list(np.where(dt > 0.1)[0] + 1)
        if idxs[-1] != len(df):
            idxs += [None]  # select until the end of the list
        for k in range(len(idxs) - 1):  # Chunking into alpha bin
            sl = slice(idxs[k], idxs[k + 1], None)
            df_k = df[sl]
            _p = df_k["value"]
            _dp = df_k["value_err"]
            _isse = np.sum(1/_dp**2)  # Inverse Square Sum of Errors
            _dats_ast['label'].append(label)
            _dats_ast["reference"].append(reference)
            _dats_ast["filter"].append(filt)
            _dats_ast['midjd'].append(np.median(df_k['jd']))
            _dats_ast['alpha'].append(np.median(df_k['alpha']))
            _dats_ast["Pr"].append(np.sum(_p/_dp**2)/_isse)
            _dats_ast["dPr"].append(1/np.sqrt(_isse))
    else:  # single data point
        _dats_ast['label'].append(label)
        _dats_ast["reference"].append(reference)
        _dats_ast["filter"].append(filt)
        _dats_ast['midjd'].append(df['jd'])
        _dats_ast['alpha'].append(df['alpha'])
        _dats_ast["Pr"].append(df["value"])
        _dats_ast["dPr"].append(df["value_err"])

dats_ast = pd.DataFrame.from_dict(_dats_ast)
# ---------------------------------------------------------------------------------------------------------- #

# ********************************************************************************************************** #
# *                                      EXTRACT DATA FOR CONVENIENCE                                      * #
# ********************************************************************************************************** #
alpha = dats["alpha"].to_numpy().astype(float)
polr = dats["Pr"].to_numpy().astype(float)
dpolr = dats["dPr"].to_numpy().astype(float)

dats_msi = dats[dats["obs"] == 'MSI']
dats_oth = dats[dats["obs"] != 'MSI']
alpha_msi = dats_msi["alpha"].to_numpy().astype(float)
alpha_oth = dats_oth["alpha"].to_numpy().astype(float)
polr_msi = dats_msi["Pr"].to_numpy().astype(float)
polr_oth = dats_oth["Pr"].to_numpy().astype(float)
dpolr_msi = dats_msi["dPr"].to_numpy().astype(float)
dpolr_oth = dats_oth["dPr"].to_numpy().astype(float)
# ---------------------------------------------------------------------------------------------------------- #

# ********************************************************************************************************** #
# *                       DEFINE PARAM CLASS AND DEFINE FUNCTION-DEPENDENT VARIABLES                       * #
# ********************************************************************************************************** #


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
    k1=Param('k1', -1., 1., 0.001),
    k2=Param('k2', -1., 1., 1.e-5),
    k0=Param('k0', -1., 1., 1.e-5)
    #    k1=Param('k1', 1.e-8, 1., 0.001),
    #    k2=Param('k2', 1.e-8, 1., 1.e-5),
    #    k0=Param('k0', 1.e-8, 1., 1.e-5)
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
# ---------------------------------------------------------------------------------------------------------- #


def cfit_pol(fitfunc, funcname, df, xname="alpha", yname="Pr", dyname="dPr", use_error=True,
             absolute_sigma=True, full=True):
    ''' Convenience function for the curve fitting.
    Parameters
    ----------
    fitfunc : function object
        One of the functions defined below in this script file.

    '''
    _pars = pars[funcname]
    _p0 = p0[funcname]  # initial values for the function fit
    _bounds = bounds[funcname]  # lower and upper bounds for each paramters
    _cfit_kw = dict(p0=_p0, bounds=_bounds, absolute_sigma=absolute_sigma)
    # Find the least-squares solution
    xdata = df[xname].to_numpy().astype(float)
    ydata = df[yname].to_numpy().astype(float)
    sigma = df[dyname].to_numpy().astype(float) if use_error else None
    popt, pcov = curve_fit(fitfunc, xdata, ydata, sigma=sigma, **_cfit_kw)

    if full:
        data = dict(x=xdata, y=ydata, dy=sigma)
        res = dict(pars=_pars, p0=_p0, bounds=_bounds, cfit_kw=_cfit_kw)
        return popt, pcov, data, res
    else:
        return popt, pcov


def _listify(x):
    ''' A very simple function to turn str, int, float to list while list, tuple, ndarray to list.
    Other data types are not expected (no need to make such complicated function for here...)
    '''
    if not isinstance(x, (tuple, list, np.ndarray)):
        x = [x]
    else:
        x = list(x)
    return x


def plot_data(axs, xlims=[(0, 160), (0, 35)], ylims=[(-5, 65), (-3, 4)],
              xmajlockws=[20, 10], xminlockws=[10, 5], ymajlockws=[10, 1], yminlockws=[5, 0.5],
              mkw_msi=dict(color='r', marker='o', ms=4, label="MSI"),
              mkw_oth=dict(color='g', marker='o', ms=4, mfc='none', label="Others"),
              errb_kw=dict(capsize=0, elinewidth=1, ls='')):
    ''' Utility for plotting data
    axs must be in order of ``(Axes for full data, Axes for negative branch)``.
    '''
    for i, ax in enumerate(axs.flat):
        ax.errorbar(alpha_msi, polr_msi, dpolr_msi, **errb_kw, **mkw_msi)
        ax.errorbar(alpha_oth, polr_oth, dpolr_oth, **errb_kw, **mkw_oth)
        ax.set(xlabel="Phase angle [˚]", ylabel=r"$ P_\mathrm{r} $ [%]")
        ax.axhline(0, color='k', ls=':')
        ax.set(xlim=xlims[i], ylim=ylims[i])
        yvu.linticker(ax,
                      xmajlockws=xmajlockws[i], xminlockws=xminlockws[i],
                      ymajlockws=ymajlockws[i], yminlockws=yminlockws[i])


# ********************************************************************************************************** #
# *                                 TRIGONOMETRIC FUNCTION (LUMME-MUINONEN)                                * #
# ********************************************************************************************************** #

def trigp(x, h=p0['trigp_b'][0], a0=p0['trigp_b'][1], c1=p0['trigp_b'][2], c2=p0['trigp_b'][3]):
    ''' Lumme-Muinonen function in pure python mode.
    '''
    term1 = (np.sin(x*D2R) / np.sin(a0*D2R))**c1
    term2 = (np.cos(x/2*D2R) / np.cos(a0/2*D2R))**c2
    term3 = np.sin((x - a0)*D2R)
    Pr = h / D2R * term1 * term2 * term3
    return Pr


@nb.njit  # (parallel=True)
def _nb_trigp(x, h=p0['trigp_b'][0], a0=p0['trigp_b'][1], c1=p0['trigp_b'][2], c2=p0['trigp_b'][3]):
    ''' Lumme-Muinonen function in numba mode.
    '''
    term1 = (np.sin(x*D2R) / np.sin(a0*D2R))**c1
    term2 = (np.cos(x/2*D2R) / np.cos(a0/2*D2R))**c2
    term3 = np.sin((x - a0)*D2R)
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
