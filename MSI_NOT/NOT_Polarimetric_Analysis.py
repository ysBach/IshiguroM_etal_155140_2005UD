'''
---------------
Before running
---------------

This is the code for making the "Masking image" of FITS file.
The "Masking image" masks the 1) nearby stars, 2) Cosmic ray, (if this is MSI data) 3) the polarization mask area.

1.
Input file:  '*.fits'                      Preprocessed FITS file
             '*.mag.1'                     IRAF Phot file containing target's center info.
                                           See below (i.e.,2. What you need to run this code)
             'mask_*.fits'                 Masking image produced by 'Masking_image.py'.

Output file: 'result_Photo_*.csv'          Photometric result of each images
             'result_Pol_*.csv'            Polarimetric result of each sets

2.What you need to run this code.
  (The following packages must be installed.)
  i.  astropy (https://www.astropy.org/)
  iii.``*.mag.1`` file from IRAF's Phot package that contains the center of the target (2005 UD).
      The first line should be the center in the ordinary component, and the
      second line should be for extraordinary component.
      We've also uploaded the ``*.mag.1`` file that we used with this code.


3.
In this code, the center of the target is found by using the phot task of IRAF.
So, we need the ``.mag`` file to import the coordinate of target's ceter.
There is no problem if you find the target's center by other methods.
All you need to do is modifying the part that brings the central coordinate of target.
See ``BRING THE CENTER COORDINATE OF 2005 UD`` part.

4.
Directory should contain the complete sets consist of 4 images (taken at HWP=0, 22.5, 45, 67.5 deg).
If even one set does not have 4 images (e.g., set having images taken at HWP = 0, 45, 67.5 deg),
an error will occur.
'''

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

from astropy.io import fits, ascii
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import gaussian_fwhm_to_sigma

mpl.rc('figure', max_open_warning=0)


# ********************************************************************************************************** #
# *                                           INPUT VALUE                                                  * #
# ********************************************************************************************************** #


Observatory = 'Z23'  # Observatory code of NOT
OBJECT = 155140  # 2005 UD

path = os.path.join('Directory/Where/image/to/be/analyzed/is/saved/')  # Path of directory where images to be analyzed are saved
IMAGE_list = glob.glob(os.path.join(path, '*.fits'))  # Bring all FITS preprocessed image in the directory

####################################
# Calibration Parameter
####################################
eff = 1  # Polarimetric efficiency of instrument
eff_err = 0  # Error of polarimetric efficiency of instrument

q_inst = -0.000428  # Instrumental polarization, q_{inst}
u_inst = -0.000765  # Instrumental polarization, u_{inst}
eq_inst = 0.000651  # Error of instrumental polarization, error of q_{inst}
eu_inst = 0.000745  # Error of instrumental polarization, error of u_{inst}

theta_offset = 93.10  # Offset of polarization angle
err_theta = 0.0565  # Error of offset

####################################
# Photometry Parameter
####################################
Aperture_scale = 4.3    # Aperture radius = Aperture_scale * FWHM
ANN_sacle = 8         # Annulus radius = ANN_sacle * FWHM
Dan = 15          # [pix] #Dannulus size

# ********************************************************************************************************** #


Photo_Log = pd.DataFrame({})
Pol_Log = pd.DataFrame({})
order = np.arange(0, len(IMAGE_list), 4)
IMAGE_list = sorted(IMAGE_list)
for z in order:
    print('==== Set {0:03d} ===='.format(int(z/4+1)))
    SET = [IMAGE_list[z], IMAGE_list[z+1], IMAGE_list[z+2], IMAGE_list[z+3]]  # Image taken at HWP=0,22.5,45,67.5

    kappa = []
    err_kappa = []
    PSANG = []
    PA = []
    JD_mean = []
    aper_seri_o = ''
    aper_seri_e = ''
    snr = 0
    file_name = SET[0].split('/')[-1].split('.')[0] + ' ~ '+SET[3].split('/')[-1][-7:]

    for ang in range(0, 4):
        image_i = SET[ang]
        hdul = fits.open(image_i)
        header = hdul[0].header
        data = hdul[0].data
        epoch = Time(header['DATE-OBS'], format='isot')
        epoch_jd = Time(epoch, format='isot', scale='utc').jd
        JD_mean.append(epoch_jd)
        exp = header['EXPTIME']
        try:
            header['GAIN']
        except KeyError:
            print('There is no GAIN in header. Using gain 0.16 & RN = 4.3')
            gain = 0.16
            RN = 4.3
        else:
            gain = header['GAIN']
            RN = header['RDNOISE']

        # ************************************************************************************* #
        # *                Bring the observer quantities from JPL Horizons                    * #
        # ************************************************************************************* #
        obj = Horizons(id=OBJECT, location=Observatory, epochs=epoch_jd)
        eph = obj.ephemerides()
        psANG = eph['sunTargetPA'][0]  # [deg]
        pA = eph['alpha'][0]  # [deg]
        PSANG.append(psANG)
        PA.append(pA)

        # ************************************************************************************* #
        # *                   BRING THE CENTER COORDINATE OF 2005 UD                          * #
        # ************************************************************************************* #
        # Here, we use Phot package in IRAF to find the center of the target
        # If you find the center of the target by other methods, please change this part.

        # Bring the center
        mag1 = ascii.read(image_i+'.mag.1')
        xo, yo = (mag1['XCENTER'][0]-1, mag1['YCENTER'][0]-1)  # pixel coordinate (x,y) of target's center in ordinary component
        xe, ye = (mag1['XCENTER'][1]-1, mag1['YCENTER'][1]-1)  # pixel coordinate (x,y) of target's center in extra. component

        # ************************************************************************************* #
        # *                            Determine FWHM of target                               * #
        # ************************************************************************************* #

        ##############################
        #  Ordinary
        ##############################

        y_1, y_2 = int(yo-20), int(yo+20)
        x_1, x_2 = int(xo-20), int(xo+20)
        crop_o = data[y_1:y_2, x_1:x_2]  # Crop the image based on the target
        crop_o = crop_o - np.median(crop_o)
        y, x = np.mgrid[:len(crop_o), :len(crop_o[0])]
        g_init = Gaussian2D(x_mean=20, y_mean=20,
                            theta=0,
                            amplitude=crop_o[20, 20],
                            bounds={'x_mean': (18, 22),
                                    'y_mean': (18, 22)})

        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x, y, crop_o)
        center_x = fitted.x_mean.value
        center_y = fitted.y_mean.value
        fwhm_o = max(fitted.x_fwhm, fitted.y_fwhm)
        aper_seri_o += '{0:.1f} '.format(fwhm_o)

        ##############################
        #  Extra-Ordinary
        ##############################
        y_1, y_2 = int(ye-20), int(ye+20)
        x_1, x_2 = int(xe-20), int(xe+20)
        crop_e = data[y_1:y_2, x_1:x_2]
        crop_e = crop_e - np.median(crop_e)
        y, x = np.mgrid[:len(crop_e), :len(crop_e[0])]
        g_init = Gaussian2D(x_mean=20, y_mean=20,
                            theta=0,
                            amplitude=crop_e[20, 20],
                            bounds={'x_mean': (18, 22),
                                    'y_mean': (18, 22)})

        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x, y, crop_e)
        center_x = fitted.x_mean.value
        center_y = fitted.y_mean.value
        fwhm_e = max(fitted.x_fwhm, fitted.y_fwhm)
        aper_seri_e += '{0:.1f} '.format(fwhm_e)

        # ************************************************************************************* #
        # *                            Bring the Masking Image                                * #
        # ************************************************************************************* #
        def circle_masking(image, x1, y1, r):

            Masking_image = np.zeros(np.shape(image))
            for yi in range(len(image)):
                for xi in range(len(image)):
                    if (xi - x1)**2 + (yi-y1)**2 < (r)**2:
                        Masking_image[yi, xi] = 1
            return Masking_image
        masking_file = os.path.join(path, 'Masking', 'mask_'+image_i.split('/')[-1])
        if os.path.exists(masking_file) == True:
            hdul_mask = fits.open(os.path.join(path, 'Masking', 'mask_'+image_i.split('/')[-1]))[0]
            masking = hdul_mask.data
        else:
            print('There is no '+os.path.join(path, 'Masking', 'mask_'+image_i.split('/')[-1]))
            print('Continue without Masking image.')
            masking = np.zeros(np.shape(data))
        masking = (masking).astype(bool)
        masking_o = np.copy(masking)  # Masking image for ordinary component
        masking_e = np.copy(masking)  # Masking image for extra-ordiary component

        mask_fwhm = max(fwhm_o, fwhm_e)
        for yi in range(len(masking)):
            for xi in range(len(masking[0])):
                if (xi - xo)**2 + (yi-yo)**2 < 65*mask_fwhm :
                    masking_e[yi, xi] = 1  # Masking the ordinary component of target
                if (xi - xe)**2 + (yi-ye)**2 < 65*mask_fwhm :
                    masking_o[yi, xi] = 1  # Masking the extra-ordinary component of target

        # ************************************************************************************* #
        # *                         Re-Determine FWHM of target                               * #
        # ************************************************************************************* #
        def skyvalue(data, y0, x0, r_in, r_out, masking):
            masking = masking.astype(bool)
            ann = CircularAnnulus([x0, y0], r_in=r_in, r_out=r_out)
            phot = aperture_photometry(data, ann, mask=masking)

            phot1 = aperture_photometry(masking*1, ann)
            pixel_count = phot1['aperture_sum'][0]
            sky = phot['aperture_sum'][0]/(ann.area-pixel_count)

            masked_image = np.ma.masked_array(data, masking)

            # Determine sky std
            y_in = int(y0-r_out)
            y_out = int(y0+r_out)
            x_in = int(x0-r_out)
            x_out = int(x0+r_out)
            if y_in < 0:
                y_in = 0
            if y_out < 0:
                y_out = 0
            if x_in < 0:
                x_in = 0
            if x_out < 0:
                x_out = 0
            masked_image = masked_image[y_in:y_out, x_in:x_out]
            new_mask = np.zeros(np.shape(masked_image))+1
            for yi in range(len(masked_image)):
                for xi in range(len(masked_image[0])):
                    position = (xi - r_out)**2 + (yi-r_out)**2
                    if position < (r_out)**2 and position > r_in**2:
                        new_mask[yi, xi] = 0
            new_mask = new_mask.astype(bool)
            Sky_region = np.ma.masked_array(masked_image, new_mask)
            std = np.ma.std(Sky_region)
            return(sky, std, ann.area-pixel_count)

        def signal_to_noise(source_eps, sky_std, rd, npix):
            signal = source_eps
            noise = np.sqrt( (source_eps  + npix *
                              (sky_std**2 )) + npix * rd ** 2)
            return signal / noise

        ##############################
        #  Ordinary
        ##############################
        y_1, y_2 = int(yo-20), int(yo+20)
        x_1, x_2 = int(xo-20), int(xo+20)

        crop_data = data[y_1:y_2, x_1:x_2]
        sky_tem, std_tem, sky_area_tem = skyvalue(data, yo, xo, 25, 35, masking_o)  # Derive sky value
        crop_data_sub = crop_data-sky_tem
        masking2 = masking_o[y_1:y_2, x_1:x_2]
        data2 = np.ma.masked_array(crop_data_sub, masking2)
        y, x = np.mgrid[:len(crop_data), :len(crop_data[0])]
        g_init = Gaussian2D(x_mean=20, y_mean=20,
                            theta=0,
                            amplitude=crop_o[20, 20],
                            bounds={'x_mean': (18, 22),
                                    'y_mean': (18, 22)})

        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x, y, data2)
        center_x = fitted.x_mean.value
        center_y = fitted.y_mean.value
        std_aper = max([fitted.x_stddev.value, fitted.y_stddev.value])

        re_g_init = Gaussian2D(amplitude=fitted.amplitude.value,
                               x_mean=center_x,
                               y_mean=center_y,
                               x_stddev=fitted.x_stddev.value,
                               y_stddev=fitted.y_stddev.value)
        fitter = LevMarLSQFitter()
        fitted = fitter(re_g_init, x, y, data2)
        FWHM_AP_ordi = max([re_g_init.x_fwhm, re_g_init.y_fwhm])
        amp_o = re_g_init.amplitude.value

        ##############################
        #  Extra-Ordinary
        ##############################
        y_1, y_2 = int(yo-20), int(yo+20)
        x_1, x_2 = int(xo-20), int(xo+20)

        crop_data = data[y_1:y_2, x_1:x_2]
        sky_tem, std_tem, sky_area_tem = skyvalue(data, ye, xe, 25, 35, masking_e)
        crop_data_sub = crop_data-sky_tem
        masking2 = masking_e[y_1:y_2, x_1:x_2]
        data2 = np.ma.masked_array(crop_data_sub, masking2)
        y, x = np.mgrid[:len(crop_data), :len(crop_data[0])]
        g_init = Gaussian2D(x_mean=20, y_mean=20,
                            theta=0,
                            amplitude=crop_o[20, 20],
                            bounds={'x_mean': (18, 22),
                                    'y_mean': (18, 22)})

        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x, y, data2)
        center_x = fitted.x_mean.value
        center_y = fitted.y_mean.value
        std_aper = max([fitted.x_stddev.value, fitted.y_stddev.value])

        re_g_init = Gaussian2D(amplitude=fitted.amplitude.value,
                               x_mean=center_x,
                               y_mean=center_y,
                               x_stddev=fitted.x_stddev.value,
                               y_stddev=fitted.y_stddev.value)
        fitter = LevMarLSQFitter()
        fitted = fitter(re_g_init, x, y, data2)
        FWHM_AP_extra = max([re_g_init.x_fwhm, re_g_init.y_fwhm])
        amp_e = re_g_init.amplitude.value

        # ************************************************************************************* #
        # *                        Circular Aperture Photometry                               * #
        # ************************************************************************************* #
        def signal_to_noise(source_eps, sky_std, rd, npix):
            signal = source_eps
            noise = np.sqrt( (source_eps  + npix *
                              (sky_std**2 )) + npix * rd ** 2)
            return signal / noise

        FWHM_sel = max(FWHM_AP_ordi, FWHM_AP_extra)
        Aperture_radius = Aperture_scale*FWHM_sel/2
        Ann = ANN_sacle*FWHM_sel/2
        Ann_out = Ann+Dan

        # Determine sky value by aperture
        Aper_o = CircularAperture([xo, yo], Aperture_radius)  # Set aperture
        sky_o, sky_std_o, area_o = skyvalue(data, yo, xo, Ann, Ann_out, masking_o)  # Set area determinung Sk #[count]

        Aper_e = CircularAperture([xe, ye], Aperture_radius)  # Set aperture
        sky_e, sky_std_e, area_e = skyvalue(data, ye, xe, Ann, Ann_out, masking_e)  # Set area determinung Sk

        sky_std_o = sky_std_o*gain
        sky_std_e = sky_std_e*gain

        Flux_o = aperture_photometry(data - sky_o, Aper_o, masking)['aperture_sum'][0]*gain
        Sum_o = aperture_photometry(data, Aper_o)['aperture_sum'][0]*gain
        ERR_o = np.sqrt(Flux_o + 3.14*Aperture_radius**2*(sky_std_o**2 + (RN*gain)**2))
        Snr_o = signal_to_noise(Flux_o, sky_std_o, RN, Aperture_radius**2*3.14)

        Flux_e = aperture_photometry(data - sky_e, Aper_e, masking)['aperture_sum'][0]*gain
        Sum_e = aperture_photometry(data, Aper_e)['aperture_sum'][0]*gain
        ERR_e = np.sqrt(Flux_e + 3.14*Aperture_radius**2*(sky_std_e**2 + (RN*gain)**2))
        Snr_e = signal_to_noise(Flux_e, sky_std_e, RN, Aperture_radius**2*3.14)
        snr += Snr_o + Snr_e
        kappa.append(Flux_e/Flux_o)
        err_kappa.append( (Flux_e/Flux_o**2 * ERR_o)**2
                          + (1/Flux_o * ERR_e)**2)

        # ************************************************************************************* #
        # *                       Plot the aperture photometry                                * #
        # ************************************************************************************* #
        def circle(x, y, r):
            theta = np.linspace(0, 2*np.pi, 100)
            x1 = r*np.cos(theta)+y
            x2 = r*np.sin(theta)+x
            return(x2.tolist(), x1.tolist())
        figsize = 50
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        plot_data = np.ma.masked_array(data, masking_o)
        if abs(amp_o*0.01) > 5000:
            vmin, vmax = -5000, 5000
        else:
            vmin, vmax = -amp_o*0.01, amp_o*0.01
        vmin, vmax = -700, 700
        im = ax[0].imshow(plot_data - sky_o, vmin=vmin, vmax=vmax, cmap='seismic')
        xi, yi = circle(xo, yo, Aperture_radius)
        ax[0].plot(xi, yi, color='y', lw=4)
        xi, yi = circle(xo, yo, Ann)
        ax[0].plot(xi, yi , color='c', lw=4)
        xi, yi = circle(xo, yo, Ann_out)
        ax[0].plot(xi, yi , color='c', lw=4)
        ax[0].plot(xo, yo, marker='X', ms=8, ls='', color='b')
        ax[0].set_xlim(xo-figsize, xo+figsize)
        ax[0].set_ylim(yo-figsize, yo+figsize)
        ax[0].set_title(image_i.split('/')[-1]+' Ordi')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plot_data = np.ma.masked_array(data, masking_e)
        if abs(amp_e*0.01) > 5000:
            vmin, vmax = 0, 8000
        else:
            vmin, vmax = -amp_e*0.01, amp_e*0.01
        vmin, vmax = -700, 700
        im = ax[1].imshow(plot_data - sky_e, vmin=vmin, vmax=vmax, cmap='seismic')
        xi, yi = circle(xe, ye, Aperture_radius)
        ax[1].plot(xi, yi, color='y', lw=4)
        xi, yi = circle(xe, ye, Ann)
        ax[1].plot(xi, yi , color='c', lw=4)
        xi, yi = circle(xe, ye, Ann_out)
        ax[1].plot(xi, yi , color='c', lw=4)
        ax[1].plot(xe, ye, marker='X', ms=8, ls='', color='b')
        ax[1].set_xlim(xe-figsize, xe+figsize)
        ax[1].set_ylim(ye-figsize, ye+figsize)
        divider = make_axes_locatable(ax[1])
        ax[1].set_title(image_i.split('/')[-1]+' Extra')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

        # Save the result of photometry
        print(image_i.split('/')[-1])
        epoch = Time(header['DATE-AVG'], format='isot')
        Photo_Log = Photo_Log.append({'Object': header['OBJECT'],
                                      'Filename': image_i.split('/')[-1],
                                      'set': int(z/4)+1,
                                      'HWPANG': header['FARETANG'],
                                      'TIME': epoch,
                                      'JD': epoch.jd,
                                      'FWHM_o': FWHM_AP_ordi,
                                      'FWHM_e': FWHM_AP_extra,
                                      'Aper [pix]': Aperture_radius,
                                      'EXP [s]': exp,
                                      'Ann': Ann,
                                      'Ann_out': Ann_out,
                                      'Flux_o': Flux_o,
                                      'eFLux_o': ERR_o,
                                      'Flux_e': Flux_e,
                                      'eFLux_e': ERR_e,
                                      'SNR_o': Snr_o,
                                      'SNR_e': Snr_e,
                                      'Sky_o': sky_o,
                                      'eSky_o': sky_std_o,
                                      'Sky_e': sky_e,
                                      'eSky_e': sky_std_e}, ignore_index=True)

    # ************************************************************************************* #
    # *                         Calculate Stokes Parameter                                * #
    # ************************************************************************************* #
    PsANG_av = np.mean(PSANG)  # the average position angles of Sun-target radius vector of a set
    PA_av = np.mean(PA)  # the average phase angle of a set
    JD_av = np.mean(JD_mean)  # the average JD of a set

    k_0 = kappa[0]
    k_45 = kappa[2]
    k_22 = kappa[1]
    k_67 = kappa[3]

    ek_0 = err_kappa[0]
    ek_45 = err_kappa[2]
    ek_22 = err_kappa[1]
    ek_67 = err_kappa[3]

    aQ = np.sqrt(k_0/k_45)
    aU = np.sqrt(k_22/k_67)

    q = (1-aQ)/(1+aQ)  # Q/I
    u = (1-aU)/(1+aU)  # U/I

    q_ran = (aQ/((aQ + 1)**2)  * np.sqrt(ek_0 + ek_45))
    u_ran = (aU/((aU + 1)**2)  * np.sqrt(ek_22 + ek_67))

    # ====================
    # Correct Efficiency
    # ====================
    qq = q/eff
    uu = u/eff

    # random error of corrected q,u
    qq_ran = q_ran/eff
    uu_ran = u_ran/eff

    # the systematic errors
    qq_sys = np.abs(q)*eff_err/eff
    uu_sys = np.abs(u)*eff_err/eff

    # ====================
    # Correc Instrumental polarization
    # ====================

    qqq = qq - q_inst
    uuu = uu - u_inst

    # random error of corrected q,u
    qqq_ran = qq_ran
    uuu_ran = uu_ran

    # the systematic errors
    qqq_sys = np.sqrt( qq_sys**2 + eq_inst**2)
    uuu_sys = np.sqrt( uu_sys**2 + eu_inst**2)

    # ====================
    # Transform_CelestialCoord
    # ====================
    theta = np.deg2rad(theta_offset)
    the_err = np.deg2rad(err_theta)

    qqqq = qqq * np.cos(2*theta) - uuu*np.sin(2*theta)
    uuuu = qqq * np.sin(2*theta) + uuu*np.cos(2*theta)

    qqqq_ran = np.sqrt( (qqq_ran*np.cos(2*theta))**2 + (uuu_ran*np.sin(2*theta))**2 )
    uuuu_ran = np.sqrt( (qqq_ran*np.sin(2*theta))**2 + (uuu_ran*np.cos(2*theta))**2 )

    qqqq_sys = np.sqrt( (qqq_sys*np.cos(2*theta))**2 +
                        (uuu_sys*np.sin(2*theta))**2 +
                        (np.pi/180*2*uuuu*the_err)**2 )
    uuuu_sys = np.sqrt( (qqq_sys*np.sin(2*theta))**2 +
                        (uuu_sys*np.cos(2*theta))**2 +
                        (np.pi/180*2*qqqq*the_err)**2 )

    q, q_ran, q_sys, u, u_ran, u_sys = qqqq, qqqq_ran, qqqq_sys, uuuu, uuuu_ran, uuuu_sys

    # ************************************************************************************* #
    # *                       Calculate Polarimetric result                               * #
    # ************************************************************************************* #
    P = np.sqrt(q**2 + u**2)
    P_ran = np.sqrt( (q*q_ran)**2 + (u*u_ran)**2 )/P
    P_sys = np.sqrt( (q*q_sys)**2 + (u*u_sys)**2 )/P
    theta_pol = np.rad2deg(1/2 * np.arctan2(u, q))
    # Random noise correction (Wardle & Kronberg 1974)
    if P**2 > P_ran**2:
        print('Random error bias correction is done.')
        P_cor = np.sqrt(P**2 - P_ran**2)
    elif P**2 < P_ran**2 :
        print('Due to P < random error, P = 0% ')
        P_cor = 0

    P_error = np.sqrt(P_ran**2 + P_sys**2)  # Polarization error
    if P_cor != 0:
        ran_PolAng = 1/2 * 180/3.14 * P_ran/P_cor
        sys_PolAng = 1/2 * 180/3.14 * P_sys/P_cor
        PolAng_error = np.sqrt(ran_PolAng**2 + sys_PolAng**2)
    elif P_cor == 0:
        ran_PolAng = 51.96
        sys_PolAng = 51.96
        PolAng_error = 51.96
        # Naghizadeh-Khouei & Clarke 1993

    if PsANG_av + 90 < 180:
        pi = PsANG_av + 90
    else:
        pi = PsANG_av - 90

    # Converted a polarization degree with respect to the scattering plane
    theta_r = theta_pol - pi
    Pr = P_cor * np.cos(2*np.deg2rad(theta_r))
    Pol_Log = Pol_Log.append({'filename': file_name,
                              'Filter': 'Rc',
                              'JD': JD_av,
                              'alpha [deg]': PA_av,
                              'PsANG [deg]': PsANG_av,
                              'FWHM_o': aper_seri_o,
                              'FWHM_e': aper_seri_e,
                              'Aper_radius [pix]': Aperture_radius,
                              'q': q,
                              'u': u,
                              'ran_q': q_ran,
                              'ran_u': u_ran,
                              'sys_q': q_sys,
                              'sys_u': u_sys,
                              'PsANG': PsANG_av,
                              'theta': theta_pol,
                              'theta_r': theta_r,
                              'eTheta': PolAng_error,
                              'P': P_cor,
                              'eP': np.sqrt(P_ran**2 + P_sys**2)},
                             ignore_index=True)

# ************************************************************************************* #
# *                       Calculate the weighted mean                                 * #
# ************************************************************************************* #


def weight(x, err):
    x = np.array(x)
    err = np.array(err)

    w = 1/err**2
    sumW = np.sum(w)
    weight = w/sumW

    xav = np.sum(weight*x)
    Err = 1/np.sqrt(sumW)

    return(xav, Err)


time_ = Time(Pol_Log['JD'][0], format='jd').iso
q_av, ranq_av = weight(Pol_Log['q'], Pol_Log['ran_q'])
u_av, ranu_av = weight(Pol_Log['u'], Pol_Log['ran_u'])
sysq_av = np.mean(Pol_Log['sys_q'])
sysu_av = np.mean(Pol_Log['sys_u'])
errq_av = (ranq_av**2 + sysq_av**2)**0.5
erru_av = (ranu_av**2 + sysu_av**2)**0.5

P = np.sqrt(q_av**2+u_av**2)
ran_P = np.sqrt((q_av*ranq_av)**2 + (u_av*ranu_av)**2)/P
sys_P = np.sqrt((q_av*sysq_av)**2 + (u_av*sysu_av)**2)/P
eP = np.sqrt(ran_P**2 + sys_P**2)

# Random noise correction (Wardle & Kronberg 1974)
if P**2 - ran_P**2 < 0:
    Pcor = 0
    print('Due to P < random error, P = 0% ')
else:
    Pcor = np.sqrt(P**2 - ran_P**2)
    print('Random error bias correction is done.')

# Converted a polarization degree with respect to the scattering plane
theta = 1/2*np.rad2deg(np.arctan2(u_av, q_av))
psang = np.mean(Pol_Log['PsANG'])
if psang+90 > 180:
    pi = psang-90
else:
    pi = psang + 90

theta_r = theta - (pi)
if Pcor != 0:
    theta_ran = 1/2*180/3.14*ran_P/Pcor
    theta_sys = 1/2*180/3.14*sys_P/Pcor
    eTheta = np.sqrt(theta_ran**2 + theta_sys**2)
elif Pcor == 0:
    theta_ran = 51.96
    theta_sys = 51.96
    eTheta = 51.96
    # Naghizadeh-Khouei & Clarke 1993
Pol_Log = Pol_Log.append({'filename': 'Weighted_average',
                          'JD': np.mean(Pol_Log['JD'].values),
                          'alpha [deg]': np.mean(Pol_Log['alpha [deg]'].values),
                          'PsANG [deg]': np.mean(Pol_Log['PsANG [deg]'].values),
                          'q': q_av,
                          'u': u_av,
                          'ran_q': ranq_av,
                          'ran_u': ranu_av,
                          'sys_q': sysq_av,
                          'sys_u': sysu_av,
                          'theta': theta,
                          'theta_r': theta_r,
                          'P': Pcor,
                          'eP': eP,
                          'Aper_radius [pix]': np.mean(Pol_Log['Aper_radius [pix]'].values),
                          'Pr': P_cor*np.cos(np.deg2rad(theta_r*2)),
                          'eTheta': eTheta
                          },
                         ignore_index=True)

Pol_name = ['filename', 'JD', 'alpha [deg]', 'PsANG [deg]',
            'q', 'u', 'ran_q', 'ran_u', 'sys_q', 'sys_u',
            'P', 'eP', 'Pr', 'theta', 'theta_r', 'eTheta',
            'Aper_radius [pix]']
Phot_name = ['Object', 'Filename', 'set', 'TIME', 'JD',
             'HWPANG', 'FWHM_o', 'FWHM_e', 'EXP [s]',
             'Aper [pix]', 'Ann', 'Ann_out', 'Flux_o',
             'eFLux_o', 'Flux_e', 'eFLux_e', 'Sky_o',
             'eSky_o', 'Sky_e', 'eSky_e', 'SNR_o',
             'SNR_e']

Pol_Log = Pol_Log.reindex(columns=Pol_name)
Photo_Log = Photo_Log.reindex(columns=Phot_name)
Pol_Log = Pol_Log.round({'JD': 6, 'alpha [deg]': 2, 'PsANG [deg]': 2,
                         'q': 4, 'u': 4, 'ran_q': 4, 'ran_u': 4,
                         'sys_q': 4, 'sys_u': 4, 'theta': 2, 'theta_r': 2,
                         'P': 4, 'eP': 4, 'Aper_radius [pix]': 2, 'Pr': 4,
                         'eTheta': 2})
Photo_Log = Photo_Log.round({'HWPANG': 2, 'JD': 6, 'FWHM_o': 2, 'FWHM_e': 2,
                             'Aper [pix]': 2, 'EXP [s]': 2, 'Ann': 2,
                             'Ann_out': 2, 'Flux_o': 4, 'eFLux_o': 4,
                             'Flux_e': 4, 'eFLux_e': 4, 'SNR_o': 2,
                             'SNR_e': 2, 'Sky_o': 4, 'eSky_o': 4,
                             'Sky_e': 4, 'eSky_e': 4})
Pol_Log.to_csv(os.path.join(path, 'result_Pol_{}.csv'.format(time_.split(' ')[0])))
Photo_Log.to_csv(os.path.join(path, 'result_Photo_{}.csv'.format(time_.split(' ')[0])))
print(os.path.join(path, 'result_Pol_{}.csv'.format(time_.split(' ')[0])) + ' is created.')
print(os.path.join(path, 'result_Photo_{}.csv'.format(time_.split(' ')[0])) + ' is created.')
