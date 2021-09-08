import numpy as np

__all__ = ["hist"]


def hist(data, bins, range=None, normed=None, weights=None, density=None):
    ''' Simplify np.histogram and plt.bar/hbar.
    Parameters
    ----------
    See np.histogram

    Example
    -------
    >>> fig, ax = plt.subplots(1, 1)
    >>> pos, n, width, edges = hist(data, np.arange(-3, 3, 0.5))
    >>> ax.bar(pos, n, width, fill=False, edgecolor='k')
    '''
    n, edges = np.histogram(data, bins=bins, range=range,
                            normed=normed, weights=weights,
                            density=density)
    pos = (edges[1:] + edges[:-1])/2
    width = np.ediff1d(edges)
    return pos, n, width, edges
