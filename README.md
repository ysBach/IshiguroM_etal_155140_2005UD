This repo is intended to store all the archive codes and part of the results of **(155140) 2005 UD** observation at Seoul National University Astronomical Observatory 1-m telescope on 2018-10-12 & 2018-10-13, which are published in **Ishiguro M. et al. 2019** (in prep).



## Notebooks

Notebooks are made to give you how I reduced the data.

Note, however, that because I am not a well trained coder, it's messy. Especially because I don't know how to deal with the **memory leak problem** of ``matplotlib`` (or is it ``astropy``?), I had to halt the program from time to time and re-run the notebook, as the memory gets full after few hundred images were analyzed. Thus, especially the outputs in ``Photometer`` is not what you'll get if you run everythin from the beginning. Please refer to ``phot_log_all.txt`` to see all the outputs (I copied-and-pasted outputs from each run). **I know it's crazy** but this was the best way I could find (facepalm). If you know a better way, please let me know. I'll be more than happy.

| Notebook                | Explanations                                                 | link                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ``Reducer``             | preprocessing, CR rejection, and WCS implementation.         | [link](https://nbviewer.jupyter.org/github/ysBach/IshiguroM_etal_155140_2005UD/blob/master/Reducer.ipynb) |
| ``Photometer``          | ephemerides/PS1 query, centroid stars, Pillbox Aperture photometry, zeropoint fitting, target Circular Aperture photometry, standardization, and reduced magnitude calculation. | [link](https://nbviewer.jupyter.org/github/ysBach/IshiguroM_etal_155140_2005UD/blob/master/Photometer.ipynb) |
| ``latex_generator``     | To automatically generate the LaTeX report file              | [link](https://nbviewer.jupyter.org/github/ysBach/IshiguroM_etal_155140_2005UD/blob/master/latex_generator.ipynb) |
| ``light_curve_plotter`` | Simple plotting                                              | [link](https://nbviewer.jupyter.org/github/ysBach/IshiguroM_etal_155140_2005UD/blob/master/light_curve_plotter.ipynb) |



## Explanations on the files

* **``phot_targ_vis_inspected.csv``: Which you may be interested in. See below.**
* ``phot_targ.csv`` and ``vis_insp.csv``: original photometry and visual inspection list.
* ``summary_reduced_all.csv``: reduced files explanation for my purpose.
* ``ephem.csv``: JPL HORIZONS ephemerides.
* [SNUO1Mpy](https://github.com/ysBach/SNUO1Mpy), [ysfitsutilpy](https://github.com/ysBach/ysfitsutilpy), and [ysphotutilpy](https://github.com/ysBach/ysphotutilpy) : The frozen snapshots of my personal packages. All these are publicly available via GitHub, but I recommend you to use the frozen in versions to completely reproduce my results. I used ``photutils`` v0.7. Due to some internal API issue with ``PillBoxAperture`` object, you cannot use ``photutils`` version lower than this...
* ``Reducer.ipynb``, ``Reducer.html``: preprocessing, CR rejection, and WCS implementation.
* shell scripts for astrometry (auto-generated by ``Reducer``)
* ``Photometer.ipynb``, ``Photometer.html``, ``phot_log_all.txt``: ephemerides/PS1 query, centroid stars, Pillbox Aperture photometry, zeropoint fitting, target Circular Aperture photometry, standardization, and reduced magnitude calculation. All the logs are saved in ``phot_log_all.txt`` (I did it in a rather manual copy-and-paste manner).
* ``latex_generator.ipynb``: To automatically generate the LaTeX report file
* ``light_curve_plotter.ipynb``: Simple plotting



The following files

1. ``2005UD_photometry_Interpretation.ppdx/pdf``

   * The PDF/PPTX files which explains how I did the photometry uwing pill-box aperture. There were some "failing" cases, and I tried to explain those as honestly as possible.
   * For written explanation, please see below.

2. ``report.pdf`` (1.1GB)

   - The pdf file which contains all the information of field stars, our target, and the zero-point fitting.

3. ``target_and_zeropoint_only.pdf`` (92MB)

   * Same as report but only the target and zeropoint images. You may check what "visual inspections" I did by comparing "visually rejected images" names and figures in this file.

   

## ``phot_targ_vis_inspected.csv`` explained

``phot_targ_vis_inspected.csv`` will be the one you are interested in. The columns include

| column                                                       | explanation                                                  | comments                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| (all of the columns explained [here](https://astroquery.readthedocs.io/en/latest/api/astroquery.jplhorizons.HorizonsClass.html#astroquery.jplhorizons.HorizonsClass.ephemerides)) | The columns from JPL HORIZONS ephemerides.                   | Queried by ``astroquery.jplhorizons``                        |
| **``jd_target``**                                            | **The lighttime corrected UT time on the object at the mid-exposure time.** |                                                              |
| ``trail``                                                    | The trail length from ephemerides in pixel unit. Derived from the ephemerides, *not* from the image analysis. |                                                              |
| ``xcentroid``, ``ycentroid``                                 | Gaussian2D fitted centroid of the object.                    |                                                              |
| `` fwhm``, ``fwhm_arcsec``                                   | The seeing FWHM estimated from field stars in pixel and arcsec units*. |                                                              |
| ``n_stars``                                                  | number of stars used for the FWHM estimation and zero-point calculation. |                                                              |
| ``isnear``                                                   | if there is any PS1 object near the target (critical separation = ``6*FWHM``) | Using DAOGROUP algorithm implemented to ``photutils``        |
| ``xcenter``, ``ycenter``                                     | just duplicate of ``xcentroid``, ``ycentroid``               |                                                              |
| ``aperture_sum``, ``aperture_sum_err``                       | The aperture sum and its error (including Poisson, read noise, and 1% uncertainty from flat error, but not the Poisson term from dark frames) |                                                              |
| ``msky``, ``ssky``                                           | the estimated modal value of the sky (using SExtractor estimator formula, which is a modified version of Doodson A 1917, Biometrika) and sample standard deviation from 3-sigma 5-iteration clippings |                                                              |
| ``nrej``, ``nsky``                                           | The number of rejected sky pixels from the above clipping, and the final number of sky pixels used for sky estimation |                                                              |
| ``source_sum``, ``source_sum_err``                           | The sky-subtracted version of the aperture sum and its error (on top of ``aperture_sum_err``, sky estimation error (i.e., 1-sigma CI of ``msky``) and sky standard deviation noise are propagated as people do in IRAF) |                                                              |
| ``mag``, ``merr``                                            | The instrumental mag and its error (assuming 1st order Taylor expansion approximation using ``source_sum_err``) |                                                              |
| ``zp_fit``, ``dzp``                                          | The zero point and its error estimated from ``n_stars`` field stars using weighted mean method. | math: weighted mean/error = minimum chi^2 fit and error.     |
| ``mzp``, ``zpStd``                                           | The zero point as ``zp_fit`` but using simple mean (no weighting) and its sample standard deviation. ``zpStd`` is not included in the ``dm_red`` below. |                                                              |
| ``m_std``                                                    | Thus standardized magnitude (its error is not shown because it's identical to ``dm_red``) |                                                              |
| **``m_red``, ``dm_red``**                                    | **The reduced magnitude and its error (ontop of ``merr``, ``dzp`` is propagated as 1st order approximation)** | ``mzp`` and ``zpStd`` are not used (``zp_fit`` and ``dzp`` are used) |
| ``seroius``                                                  | 0 = no problem visible; 1 = something happened but benevolent (faint star in the annulus but not aperture, etc); 2 = seriously affected (field object in aperture) |                                                              |
| ``vis_insp_comment``                                         | ``A`` = Strange centroiding, ``B`` = PS1-uncatalogued object nearby, ``C`` = PS1-catalogued object nearby (``isnear`` is 1) |                                                              |



## Data Analysis Strategy

1. **Centroiding**. I haven't yet found perfect algorithm to do centroiding, and I guess IRAF has some highly subjectively tuned internal algorithm for the centroiding which I haven't found. From short experience, I found elliptical Gaussian fitting for a centroid box of ``cbox = 6 * FWHM_initial_guess`` with minimum value subtraction (``data - data.min()``) works reasonably. For each iteration, new Gaussian fitting is done for ``cbox``-sized box, and the centroid position is updated. If the distance to the new centroid from the old one is larger than 1 FWHM, move only 1 FWHM to that direction. Halt the iteration when shift is << 1 pixel, and iteration is done up to 10 times only.

2. **FWHM**. It's difficult to find the true FWHM from non-sidereal tracking mode, since stars are trailed. What I did is, after the centroiding to the field stars (PS1 catalogued stars) as described above, get the sigma_y (minor axis) values of the fittings, and get sigma-clipped average of that. And regard ``2*sqrt(2)*sigma_y`` as the FWHM. In reality, trailed PSF (e.g., trailed Gaussian) is rather pill-box shaped and thus the Gaussian fit may be different from the real one.

3. **Target Aperture**. For target, I used circular ``r_ap = 1.75*FWHM`` aperture. The sky annulus is defined as inner and outer radii by 4 and 6 FWHM, respectively. Normally it is recommended by classical IRAF to use ``r_ap = max(3, f_ap*FWHM)`` for ``f_ap=1.5`` to ``3`` depending on the user's preference. The reason for minimum 3 pixel radius is because IRAF uses approximated aperture sum and if ``r_ap < 3``, the non-random error in aperture photometry gets larger than about 0.1 mag (for 1 pixel radius, it gets up to 0.5 mag error). As python or IDL codes has no such problem, I didn't put lower limit on aperture. Also 1.75 is selected by my experience using the SNUO 1m telescope.

4. **Star Aperture**. Since stars are trailed, I developed a code to do *pill-box aperture* photometry. It consists of two half-ellipses (a=major, b=minor, theta=angle) with one rectangle (width=trail length, height=b of the ellipse, theta=theta of the ellipse). The theta is obtained from the sigma-clipped mean value of 2D Gaussian fits as I used in FWHM estimation above. The trail length is simply obtained from ephemerides, and a = b = 1.75 FWHM is used. If you see the aperture overplotted on stellar objects, you realize it is too large compared to visually acceptable aperture. I intended this because the tracking error sometimes make the stellar PSF be U- or V-shaped, so non-negligible stellar fluxes were lost for some images while target flux does not. Thus, sacrifising the photometric accuracy of stellar photometry, I enlarged aperture sizes.

5. **Catalog Magnitude**. The PS1 query was done by VizieR (only has PS1 DR1 data, though DR2 is already available). The conversion from PS1 filter to Johnson-Cousins R filter was adopted from Tonry et al. 2012. Only stars with PS1 r-mag between 10-15.2 were used with n_observation in PS1 >= 3(r), 3(g), 1(i). Extended objects were removed by one of the simplest method recommended by Pan-STARRS. 

   > For details which can be subjective, please see the ``Photometer`` in the zip file and the source codes in the package ``ysphotutilpy`` (``ysphotutilpy.queryutil.PanSTARRS1`` and its methods ``drop_for_diff_phot``, ``select_filters``, ``check_nearby``, ``drop_star_groups``. I tried to explain as detailed as possible in the source codes.)

6. **Standardization**. The standardization was done by comparing the R-instrumental mag (R_inst) and catalog mag (R_cat) of the field stars. The linearity curve (R_inst VS R_cat) and its residual, as well as the color-term plot (R_cat - R_inst VS g_PS1 - r_PS1) were used to visually inspect whether the stars were selected well.

7. **Visual Inspection**. It's virtually impossible to look at each image thoroughly. Thus, I let my source code to draw "report figure" for each star and target. Then automatically generate report LATEX files and compile them. While looking at the generated report PDF file, I found some images failed in correct centroiding or some coding bug, because (1) my code is not perfect yet, (2) objects were too faint to be fitted correctly, etc. This kind of "error-rate" is which I should resolve in the future, but for now I just ignored them (2 images had unexpected serious problems which I cannot find reason).

8. **Data Goodness**. For the light curve below, "good" means (1) visually not affected at all (even ``serious = 1`` are removed) and (2) ``zpStd < 0.04`` is used as just a heuristic criterion.



If you need, I can provide all the data (original, combined calibration frames, reduced object frames, etc) and intermediate products for your reference. It's already uploaded to my personal web cloud.



And if you have any question or request, please don't hesitate to contact me.



Yoonsoo P. Bach, dbstn95@gmail.com