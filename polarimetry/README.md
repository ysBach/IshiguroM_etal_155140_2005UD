# Polarimetry

The polarimetric data analysis and output files.

Each notebook itself contains most of the required instructions and explanations.
It can be rendered via ``nbviewer`` service.

The contents are

| Notebook, Script, Directory                                  | Explanation                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [``pol_analyzer.ipynb``](https://nbviewer.jupyter.org/github/ysBach/IshiguroM_etal_155140_2005UD/blob/master/polarimetry/pol_analyzer.ipynb) | Least-square and MCMC simulation based on the polarimetric data. |
| [``pol_analyzer-linexp.ipynb``](https://nbviewer.jupyter.org/github/ysBach/IshiguroM_etal_155140_2005UD/blob/master/polarimetry/pol_analyzer-linexp.ipynb) | Same as above, but using the 3-parameter linear-exponential function. Added during the first revision phase. |
| [``Phase_Curve.ipynb``](https://nbviewer.jupyter.org/github/ysBach/IshiguroM_etal_155140_2005UD/blob/master/polarimetry/Phase_Curve.ipynb) | Code used to draw the phase curve (polarimetric phase curve) |
| [``Comparison_with_others.ipynb``](https://nbviewer.jupyter.org/github/ysBach/IshiguroM_etal_155140_2005UD/blob/master/polarimetry/Comparison_with_others.ipynb) | Code used to draw the plots against the laboratory experiment data. |
| ``polutil2005ud.py``                                         | Script containing utility functions used throughout the notebooks |
| ``ysvisutilpy2005ud``                                        | Frozen snapshot of ``ysvisutilpy`` packaged used in the above notebooks (hard copied for archieval purposes). |

The resulting figures are in ``figs/`` and ``fits-below45deg``. The latter is obtained by using only the data at α<45˚, and is added during the first revision phase (as the reviewer requested to test if the results are robust against subsample at small phase angles and different functional forms for the phase curve).
