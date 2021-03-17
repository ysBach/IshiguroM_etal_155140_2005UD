from warnings import warn

import matplotlib.dates as mdates
import numpy as np
from dateutil.rrule import (DAILY, HOURLY, MINUTELY, MONTHLY, SECONDLY,
                            YEARLY)
from matplotlib.ticker import (FormatStrFormatter, LogFormatterSciNotation,
                               LogLocator, MultipleLocator, NullFormatter)

__all__ = ["colorbaring", "mplticker", "ax_tick",
           "linticker", "logticker", "logxticker", "logyticker",
           "linearticker", "append_xdate"]


def colorbaring(fig, ax, im, fmt="%.0f", orientation='horizontal',
                formatter=FormatStrFormatter, **kwargs):
    cb = fig.colorbar(im, ax=ax, orientation=orientation,
                      format=FormatStrFormatter(fmt), **kwargs)

    return cb


def ax_tick(ax, x_vals=None, x_show=None, y_vals=None, y_show=None):
    # if (x_vals is None) ^ (x_show is None):
    #     raise ValueError("All or none of x_vals and x_show should be given.")
    # if (y_vals is None) ^ (y_show is None):
    #     raise ValueError("All or none of y_vals and y_show should be given.")

    if x_vals is not None:
        x_vals = np.array(x_vals)
        if x_show is None:
            x_ticks = x_vals.copy()
            x_show = x_vals.copy()
        else:
            x_ticks = np.array([np.where(x_vals == v)[0][0] for v in x_show])

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_show)

    if y_vals is not None:
        y_vals = np.array(y_vals)
        if y_show is None:
            y_ticks = y_vals.copy()
            y_show = y_vals.copy()
        else:
            y_ticks = np.array([np.where(y_vals == v)[0][0] for v in y_show])

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_show)

    return ax


def linearticker(ax_list, xmajlocs, xminlocs, ymajlocs, yminlocs,
                 xmajfmts, ymajfmts,
                 xmajlocators=MultipleLocator,
                 xminlocators=MultipleLocator,
                 ymajlocators=MultipleLocator,
                 yminlocators=MultipleLocator,
                 xmajformatters=FormatStrFormatter,
                 ymajformatters=FormatStrFormatter,
                 majgridkw=dict(ls='-', alpha=0.8),
                 mingridkw=dict(ls=':', alpha=0.5)):
    warn("Use linticker instead of linearticker.")
    _ax_list = np.atleast_1d(ax_list)
    n_axes = len(_ax_list)
    xmajloc = np.atleast_1d(xmajlocs)
    xminloc = np.atleast_1d(xminlocs)
    ymajloc = np.atleast_1d(ymajlocs)
    yminloc = np.atleast_1d(yminlocs)
    xmajfmt = np.atleast_1d(xmajfmts)
    ymajfmt = np.atleast_1d(ymajfmts)
    if xmajloc.shape[0] != n_axes:
        xmajloc = np.repeat(xmajloc, n_axes)
    if xminloc.shape[0] != n_axes:
        xminloc = np.repeat(xminloc, n_axes)
    if ymajloc.shape[0] != n_axes:
        ymajloc = np.repeat(ymajloc, n_axes)
    if yminloc.shape[0] != n_axes:
        yminloc = np.repeat(yminloc, n_axes)
    if xmajfmt.shape[0] != n_axes:
        xmajfmt = np.repeat(xmajfmt, n_axes)
    if ymajfmt.shape[0] != n_axes:
        ymajfmt = np.repeat(ymajfmt, n_axes)

    if not isinstance(xmajlocators, (tuple, list, np.ndarray)):
        xmajlocators = [xmajlocators] * n_axes
    if not isinstance(xminlocators, (tuple, list, np.ndarray)):
        xminlocators = [xminlocators] * n_axes
    if not isinstance(ymajlocators, (tuple, list, np.ndarray)):
        ymajlocators = [ymajlocators] * n_axes
    if not isinstance(yminlocators, (tuple, list, np.ndarray)):
        yminlocators = [yminlocators] * n_axes
    if not isinstance(xmajformatters, (tuple, list, np.ndarray)):
        xmajformatters = [xmajformatters] * n_axes
    if not isinstance(ymajformatters, (tuple, list, np.ndarray)):
        ymajformatters = [ymajformatters] * n_axes

    for i, aa in enumerate(_ax_list):
        aa.xaxis.set_major_locator(xmajlocators[i](xmajloc[i]))
        aa.yaxis.set_major_locator(ymajlocators[i](ymajloc[i]))
        aa.xaxis.set_minor_locator(xminlocators[i](xminloc[i]))
        aa.yaxis.set_minor_locator(yminlocators[i](yminloc[i]))
        aa.xaxis.set_major_formatter(xmajformatters[i](xmajfmt[i]))
        aa.yaxis.set_major_formatter(ymajformatters[i](ymajfmt[i]))
        aa.grid(which='major', **majgridkw)
        aa.grid(which='minor', **mingridkw)


def _check(obj, name, n, dates=False):
    arr = np.atleast_1d(obj)
    n_arr = arr.shape[0]
    if n_arr not in [1, n]:
        raise ValueError(f"{name} must be a single object or a 1-d array"
                         + f" with the same length as ax_list ({n}).")
    else:
        newarr = arr.tolist() * (n//n_arr)

    return newarr


def _setter(setter, Setter, kw):
    # don't do anything if obj (Locator or Formatter) is None:
    if (Setter is not None) and (kw is not None):
        # matplotlib is so poor in log plotting....
        if (Setter == LogLocator) and ("numticks" not in kw):
            kw["numticks"] = 50

        if isinstance(kw, dict):
            setter(Setter(**kw))
        else:  # interpret as ``*args``
            setter(Setter(*(np.atleast_1d(kw).tolist())))
        # except:
        #     raise ValueError("Error occured for Setter={} with input {}"
        #                      .format(Setter, kw))


def mplticker(ax_list,
              xmajlocators=None, xminlocators=None,
              ymajlocators=None, yminlocators=None,
              xmajformatters=None, xminformatters=None,
              ymajformatters=None, yminformatters=None,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws=None,
              ymajfmtkws=None, yminfmtkws=None,
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    ''' Set tickers of Axes objects.
    Note
    ----
    Notation of arguments is <axis><which><name>. <axis> can be ``x`` or
    ``y``, and <which> can be ``major`` or ``minor``.
    For example, ``xmajlocators`` is the Locator object for x-axis
    major.  ``kw`` means the keyword arguments that will be passed to
    locator, formatter, or grid.
    If a single object is given for locators, formatters, grid, or kw
    arguments, it will basically be copied by the number of Axes objects
    and applied identically through all the Axes.

    Parameters
    ----------
    ax_list : Axes or 1-d array-like of such
        The Axes object(s).

    locators : Locator, None, list of such, optional
        The Locators used for the ticker. Must be a single Locator
        object or a list of such with the identical length of
        ``ax_list``.
        If ``None``, the default locator is not touched.

    formatters : Formatter, None, False, array-like of such, optional
        The Formatters used for the ticker. Must be a single Formatter
        object or an array-like of such with the identical length of
        ``ax_list``.
        If ``None``, the default formatter is not touched.
        If ``False``, the labels are not shown (by using the trick
        ``FormatStrFormatter(fmt="")``).

    grids : bool, array-like of such, optinoal.
        Whether to draw the grid lines. Must be a single bool object or
        an array-like of such with the identical length of ``ax_list``.

    lockws : dict, array-like of such, array-like, optional
        The keyword arguments that will be passed to the ``locators``.
        If it's an array-like but elements are not dict, it is
        interpreted as ``*args`` passed to locators.
        If it is (or contains) dict, it must be a single dict object or
        an array-like object of such with the identical length of
        ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    fmtkws : dict, str, list of such, optional
        The keyword arguments that will be passed to the ``formatters``.
        If it's an array-like but elements are not dict, it is
        interpreted as ``*args`` passed to formatters.
        If it is (or contains) dict, it must be a single dict object or
        an array-like object of such with the identical length of
        ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    gridkw : dict, list of such, optional
        The keyword arguments that will be passed to the grid. Must be a
        single dict object or a list of such with the identical length
        of ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    '''
    _ax_list = list(np.atleast_1d(ax_list).flatten())
    n_axes = len(_ax_list)

    _xmajlocators = _check(xmajlocators, "xmajlocators", n_axes)
    _xminlocators = _check(xminlocators, "xminlocators", n_axes)
    _ymajlocators = _check(ymajlocators, "ymajlocators", n_axes)
    _yminlocators = _check(yminlocators, "yminlocators", n_axes)

    _xmajformatters = _check(xmajformatters, "xmajformatters ", n_axes)
    _xminformatters = _check(xminformatters, "xminformatters ", n_axes)
    _ymajformatters = _check(ymajformatters, "ymajformatters", n_axes)
    _yminformatters = _check(yminformatters, "yminformatters", n_axes)

    _xmajlockws = _check(xmajlockws, "xmajlockws", n_axes)
    _xminlockws = _check(xminlockws, "xminlockws", n_axes)
    _ymajlockws = _check(ymajlockws, "ymajlockws", n_axes)
    _yminlockws = _check(yminlockws, "yminlockws", n_axes)

    _xmajfmtkws = _check(xmajfmtkws, "xmajfmtkws", n_axes)
    _xminfmtkws = _check(xminfmtkws, "xminfmtkws", n_axes)
    _ymajfmtkws = _check(ymajfmtkws, "ymajfmtkws", n_axes)
    _yminfmtkws = _check(yminfmtkws, "yminfmtkws", n_axes)

    _xmajgrids = _check(xmajgrids, "xmajgrids", n_axes)
    _xmingrids = _check(xmingrids, "xmingrids", n_axes)
    _ymajgrids = _check(ymajgrids, "ymajgrids", n_axes)
    _ymingrids = _check(ymingrids, "ymingrids", n_axes)

    _xmajgridkws = _check(xmajgridkws, "xmajgridkws", n_axes)
    _xmingridkws = _check(xmingridkws, "xmingridkws", n_axes)
    _ymajgridkws = _check(ymajgridkws, "ymajgridkws", n_axes)
    _ymingridkws = _check(ymingridkws, "ymingridkws", n_axes)

    for i, aa in enumerate(_ax_list):
        _xmajlocator = _xmajlocators[i]
        _xminlocator = _xminlocators[i]
        _ymajlocator = _ymajlocators[i]
        _yminlocator = _yminlocators[i]

        _xmajformatter = _xmajformatters[i]
        _xminformatter = _xminformatters[i]
        _ymajformatter = _ymajformatters[i]
        _yminformatter = _yminformatters[i]

        _xmajgrid = _xmajgrids[i]
        _xmingrid = _xmingrids[i]
        _ymajgrid = _ymajgrids[i]
        _ymingrid = _ymingrids[i]

        _xmajlockw = _xmajlockws[i]
        _xminlockw = _xminlockws[i]
        _ymajlockw = _ymajlockws[i]
        _yminlockw = _yminlockws[i]

        _xmajfmtkw = _xmajfmtkws[i]
        _xminfmtkw = _xminfmtkws[i]
        _ymajfmtkw = _ymajfmtkws[i]
        _yminfmtkw = _yminfmtkws[i]

        _xmajgridkw = _xmajgridkws[i]
        _xmingridkw = _xmingridkws[i]
        _ymajgridkw = _ymajgridkws[i]
        _ymingridkw = _ymingridkws[i]

        _setter(aa.xaxis.set_major_locator, _xmajlocator, _xmajlockw)
        _setter(aa.xaxis.set_minor_locator, _xminlocator, _xminlockw)
        _setter(aa.yaxis.set_major_locator, _ymajlocator, _ymajlockw)
        _setter(aa.yaxis.set_minor_locator, _yminlocator, _yminlockw)
        _setter(aa.xaxis.set_major_formatter, _xmajformatter, _xmajfmtkw)
        _setter(aa.xaxis.set_minor_formatter, _xminformatter, _xminfmtkw)
        _setter(aa.yaxis.set_major_formatter, _ymajformatter, _ymajfmtkw)
        _setter(aa.yaxis.set_minor_formatter, _yminformatter, _yminfmtkw)

        # Strangely, using ``b=_xmingrid`` does not work if it is
        # False... I had to do it manually like this... OMG matplotlib..
        if _xmajgrid:
            aa.grid(axis='x', which='major', **_xmajgridkw)
        if _xmingrid:
            aa.grid(axis='x', which='minor', **_xmingridkw)
        if _ymajgrid:
            aa.grid(axis='y', which='major', **_ymajgridkw)
        if _ymingrid:
            aa.grid(axis='y', which='minor', **_ymingridkw)


def linticker(ax_list,
              xmajlocators=MultipleLocator, xminlocators=MultipleLocator,
              ymajlocators=MultipleLocator, yminlocators=MultipleLocator,
              xmajformatters=FormatStrFormatter,
              xminformatters=NullFormatter,
              ymajformatters=FormatStrFormatter,
              yminformatters=NullFormatter,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws={},
              ymajfmtkws=None, yminfmtkws={},
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logticker(ax_list,
              xmajlocators=LogLocator, xminlocators=LogLocator,
              ymajlocators=LogLocator, yminlocators=LogLocator,
              xmajformatters=LogFormatterSciNotation,
              xminformatters=NullFormatter,
              ymajformatters=LogFormatterSciNotation,
              yminformatters=NullFormatter,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws={},
              ymajfmtkws=None, yminfmtkws={},
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logxticker(ax_list,
               xmajlocators=LogLocator, xminlocators=LogLocator,
               ymajlocators=MultipleLocator, yminlocators=MultipleLocator,
               xmajformatters=LogFormatterSciNotation,
               xminformatters=NullFormatter,
               ymajformatters=FormatStrFormatter,
               yminformatters=NullFormatter,
               xmajgrids=True, xmingrids=True,
               ymajgrids=True, ymingrids=True,
               xmajlockws=None, xminlockws=None,
               ymajlockws=None, yminlockws=None,
               xmajfmtkws=None, xminfmtkws={},
               ymajfmtkws=None, yminfmtkws={},
               xmajgridkws=dict(ls='-', alpha=0.5),
               xmingridkws=dict(ls=':', alpha=0.5),
               ymajgridkws=dict(ls='-', alpha=0.5),
               ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logyticker(ax_list,
               xmajlocators=MultipleLocator, xminlocators=MultipleLocator,
               ymajlocators=LogLocator, yminlocators=LogLocator,
               xmajformatters=FormatStrFormatter,
               xminformatters=NullFormatter,
               ymajformatters=LogFormatterSciNotation,
               yminformatters=NullFormatter,
               xmajgrids=True, xmingrids=True,
               ymajgrids=True, ymingrids=True,
               xmajlockws=None, xminlockws=None,
               ymajlockws=None, yminlockws=None,
               xmajfmtkws=None, xminfmtkws={},
               ymajfmtkws=None, yminfmtkws={},
               xmajgridkws=dict(ls='-', alpha=0.5),
               xmingridkws=dict(ls=':', alpha=0.5),
               ymajgridkws=dict(ls='-', alpha=0.5),
               ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def append_xdate(
    ax_list, xdata, ydata, concise=False, defaultfmt="%Y-%m-%d",
    kw_loc={},
    tick_year=None, tick_month=None, tick_day=None,
    tick_hour=None, tick_minute=None, tick_second=None, tick_microsecond=None,
    kw_label=dict(rotation=60, horizontalalignment='left'),
    kw_tick=dict(color='k', labelcolor='k', direction='out', length=4)
):
    '''
    xdata : tested for pandas.to_datetime() or astropy.Time.plot_date
    Uses autolocator simple method.
    I dunno why defaultfmt does not work as expected. matplotlib is too
    crazily complicated.
    '''
    locator = mdates.AutoDateLocator()

    for inter, i in zip([tick_year, tick_month, tick_day, tick_hour,
                         tick_minute, tick_second, tick_microsecond],
                        [YEARLY, MONTHLY, DAILY, HOURLY, MINUTELY, SECONDLY,
                         7]):
        if inter is not None:
            locator.intervald[i] = np.atleast_1d(inter)

    if concise:
        formatter = mdates.ConciseDateFormatter(locator, **kw_loc)
    else:
        formatter = mdates.AutoDateFormatter(locator, defaultfmt=defaultfmt,
                                             **kw_loc)

    ax_list = list(np.atleast_1d(ax_list).flatten())
    ax2_list = []
    for ax in ax_list:
        ax2 = ax.twiny()
        ax2.plot(xdata, ydata, ls='', marker='')  # Fake plot
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(formatter)
        for label in ax2.get_xticklabels():
            for k, v in kw_label.items():
                eval(f"label.set_{k}")(v)
        ax2_list.append(ax2)
    return ax2_list

    # def _locator_parse(locs):
    #     locs = np.atleast_1d(locs)
    #     news = []
    #     for loc in locs:
    #         if isinstance(loc, str):
    #             loc = loc.lower()
    #             if loc.startswith("ye"):
    #                 news.append(mdates.YearLocator)
    #             elif loc.startswith("mo"):
    #                 news.append(mdates.MonthLocator)
    #             elif loc.startswith("da"):
    #                 news.append(mdates.DayLocator)
    #             elif loc.startswith("ho") or loc.startswith("hr"):
    #                 news.append(mdates.HourLocator)
    #             elif loc.startswith("mi"):
    #                 news.append(mdates.MinuteLocator)
    #             elif loc.startswith("se"):
    #                 news.append(mdates.SecondLocator)
    #             elif loc.startswith("we"):
    #                 news.append(mdates.WeekdayLocator)
    #             # AudoDateLocator, MicrosecondLocator are ignored
    #             else:
    #                 raise ValueError("locator loc not understood")
    #         else:
    #             return news.append(loc)

    # xmajlocators = _locator_parse(xmajlocators)
    # xminlocators = _locator_parse(xminlocators)
    # if concise_date:
    #     xmajformatters=mdates.ConciseDateFormatter
