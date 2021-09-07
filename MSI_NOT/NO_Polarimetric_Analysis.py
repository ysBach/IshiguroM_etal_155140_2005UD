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
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


from astropy.io import fits, ascii
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astropy.modeling.models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import gaussian_fwhm_to_sigma

import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)
mpl.rc('figure', max_open_warning=0)


# ********************************************************************************************************** #
# *                                           INPUT VALUE                                                  * #
# ********************************************************************************************************** #


Observatory = 'q33'  # Observatory code of Pirks
OBJECT = 155140  # 2005 UD

path = os.path.join('Directory/Where/image/to/be/analyzed/is/saved/')  # Path of directory where images to be analyzed are saved
IMAGE_list = glob.glob(os.path.join(path, '*.fits'))  # Bring all FITS preprocessed image in the directory


####################################
# Photometry Parameter
####################################
Aperture_scale = 1.7   # Aperture radius = Aperture_scale * FWHM
ANN_sacle = 4         # Annulus radius = ANN_sacle * FWHM
Dan = 20              # [pix] #Dannulus size

####################################
# Calibration Parameter (Please see Ishiguro et al. 2021)
####################################
eff = 0.9913  # Polarimetric efficiency of instrument
eff_err = 0.0001  # Error of polarimetric efficiency of instrument

q_inst = 0.00791  # Instrumental polarization, q_{inst}
u_inst = 0.00339  # Instrumental polarization, u_{inst}
eq_inst = 0.00025  # Error of instrumental polarization, error of q_{inst}
eu_inst = 0.00020  # Error of instrumental polarization, error of u_{inst}

theta_offset = 3.66  # Offset of polarization angle
err_theta = 0.17  # Error of offset


# ********************************************************************************************************** #


Photo_Log = pd.DataFrame({})
Pol_Log = pd.DataFrame({})
order = np.arange(0, len(IMAGE_list), 4)
IMAGE_list = sorted(IMAGE_list)

for z in order:
    print('==== Set {0:03d} ===='.format(int(z/4+1)))
    SET = [IMAGE_list[z], IMAGE_list[z+1], IMAGE_list[z+2], IMAGE_list[z+3]]  # Image taken at HWP=0,22.5,45,67.5

    I = []  # I_ex/I_o
    S = []  # (Sigma_{I_ex}/I_ex)**2 + (Sigma_{I_o}/I_o)**2
    INR_STR = []
    INR_END = []
    PSANG = []
    PA = []
    JD_mean = []
    file_name = SET[0].split('/')[-1].split('.')[0] + ' ~ '+SET[3].split('/')[-1][-7:]
    aper_seri_o = ''
    aper_seri_e = ''
    snr = 0

    # fig,ax = plt.subplots(4,2,figsize=(20,40))
    for ang in range(0, 4):
        image_i = SET[ang]
        hdul = fits.open(image_i)
        header = hdul[0].header
        data = hdul[0].data
        RET_ANG2 = header['RET-ANG2']

        # Bring the header info.==================================================
        iNR_STR = header['INR-STR']
        iNR_END = header['INR-END']
        INST_pa = header['INST-PA']
        JD = np.mean([header['MJD-STR'], header['MJD-END']])
        JD = JD + 2400000.5  # MJD -> JD
        epoch = Time(JD, format='jd').isot
        RN = header['RDNOISE']  # [electron]
        gain = header['GAIN']  # [electron/DN]
        Filter = header['FILTER']
        exp = header['EXPTIME']  # in sec

        # ************************************************************************************* #
        # *                Bring the observer quantities from JPL Horizons                    * #
        # ************************************************************************************* #
        obj = Horizons(id=OBJECT, location=Observatory, epochs=JD)
        eph = obj.ephemerides()
        psANG = eph['sunTargetPA'][0]  # [deg]
        pA = eph['alpha'][0]  # [deg]

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
        # *                            Bring the Masking Image                                * #
        # ************************************************************************************* #
        masking_path = os.path.join(path, 'Masking', 'mask_'+image_i.split('/')[-1])
        hdul_mask = fits.open(masking_path)[0]
        masking = hdul_mask.data

        # ************************************************************************************* #
        # *                            Determine FWHM of target                               * #
        # ************************************************************************************* #
        def crop(data, row, col, size):
            row_str = int(row-size)
            row_end = int(row+size)
            col_str = int(col-size)
            col_end = int(col+size)
            data_cr = data[row_str:row_end, col_str:col_end]
            return data_cr
        ##############################
        #  Ordinary
        ##############################
        mdata = np.ma.masked_array(data, masking)
        crop_image = crop(mdata, yo, xo, 35)
        sum_crop_image = np.mean(crop_image, axis=0)
        sum_crop_image = sum_crop_image - np.mean(sum_crop_image[:10])

        g_init = Gaussian1D(amplitude=np.mean(sum_crop_image[34:36]),
                            mean=35,
                            stddev=10*gaussian_fwhm_to_sigma,
                            bounds={'mean': (33, 37),
                                    'stddev': ((5*gaussian_fwhm_to_sigma, 20*gaussian_fwhm_to_sigma))})

        x = np.arange(0, len(crop_image), 1)
        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x, sum_crop_image)

        re_g_init = Gaussian1D(amplitude=fitted.amplitude.value,
                               mean=fitted.mean.value,
                               stddev=fitted.stddev.value,
                               bounds={'mean': (33, 37),
                                       'stddev': ((5*gaussian_fwhm_to_sigma, 20*gaussian_fwhm_to_sigma))})
        fitter = LevMarLSQFitter()
        fitted = fitter(re_g_init, x, sum_crop_image)
        FWHM_ordi = 2*fitted.stddev.value*np.sqrt(2*np.log(2))
        aper_seri_o += '{0:.1f} '.format(FWHM_ordi)

        ##############################
        #  Extra-Ordinary
        ##############################
        crop_image = crop(mdata, ye, xe, 35)
        sum_crop_image = np.mean(crop_image, axis=0)
        sum_crop_image = sum_crop_image - np.mean(sum_crop_image[:10])
        g_init = Gaussian1D(amplitude=np.mean(sum_crop_image[34:36]),
                            mean=35,
                            stddev=10*gaussian_fwhm_to_sigma,
                            bounds={'mean': (33, 37),
                                    'stddev': (5*gaussian_fwhm_to_sigma, 20*gaussian_fwhm_to_sigma)})

        fitter = LevMarLSQFitter()
        fitted = fitter(g_init, x, sum_crop_image)

        re_g_init = Gaussian1D(amplitude=fitted.amplitude.value,
                               mean=fitted.mean.value,
                               stddev=fitted.stddev.value,
                               bounds={'mean': (33, 37),
                                       'stddev': (5*gaussian_fwhm_to_sigma, 20*gaussian_fwhm_to_sigma)})
        fitter = LevMarLSQFitter()
        fitted = fitter(re_g_init, x, sum_crop_image)
        FWHM_extra = 2*fitted.stddev.value*np.sqrt(2*np.log(2))
        aper_seri_e += '{0:.1f} '.format(FWHM_extra)

        # ************************************************************************************* #
        # *                        Circular Aperture Photometry                               * #
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

        FWHM_sel = max(FWHM_ordi, FWHM_extra)  # Selected FWHM
        Aperture_radius = Aperture_scale*FWHM_sel/2
        Annulus_radius = ANN_sacle*FWHM_sel/2
        Ann_out = Annulus_radius+Dan

        # Determine sky value by aperture
        Aper_o = CircularAperture([xo, yo], Aperture_radius)  # Set aperture
        sky_o, sky_std_o, area_o = skyvalue(data, yo, xo, Annulus_radius, Ann_out, masking)  # Set area determinung Sk #[count]

        Aper_e = CircularAperture([xe, ye], Aperture_radius)  # Set aperture
        sky_e, sky_std_e, area_e = skyvalue(data, ye, xe, Annulus_radius, Ann_out, masking)  # Set area determinung Sk

        sky_std_o = sky_std_o*gain
        sky_std_e = sky_std_e*gain

        ########
        masking = masking.astype(bool)
        masked_image = np.ma.masked_array(data, masking)
        y0, x0 = yo, xo
        y_in = int(y0-Ann_out)
        y_out = int(y0-Annulus_radius)
        x_in = int(x0-Ann_out)
        x_out = int(x0-Annulus_radius)
        crop_masked_images = masked_image[y_in:y_out, x_in:x_out]
        std = np.ma.std(crop_masked_images)

        # Target aperture sum
        Flux_o = aperture_photometry(data - sky_o, Aper_o, masking)['aperture_sum'][0]*gain  # in e-
        ERR_o = np.sqrt(Flux_o + 3.14*Aperture_radius**2*(sky_std_o**2 + (RN*gain)**2))  # in e-
        sky_o = sky_o*gain  # in e-
        Snr_o = signal_to_noise(Flux_o, sky_std_o, RN, Aperture_radius**2*3.14)

        Flux_e = aperture_photometry(data - sky_e, Aper_e, masking)['aperture_sum'][0]*gain  # in e-
        ERR_e = np.sqrt(Flux_e + 3.14*Aperture_radius**2*(sky_std_e**2 + (RN*gain)**2))  # in e-
        sky_e = sky_e*gain  # in e-
        Snr_e = signal_to_noise(Flux_e, sky_std_e, RN, Aperture_radius**2*3.14)
        snr += Snr_o + Snr_e

        # ************************************************************************************* #
        # *              Record the values for calculating Stokes parameter                   * #
        # ************************************************************************************* #
        I.append(Flux_e/Flux_o)
        S_ret = (ERR_e/(Flux_e))**2 + (ERR_o/(Flux_o))**2
        S.append(S_ret)
        PSANG.append(psANG)
        PA.append(pA)
        JD_mean.append(JD)
        INR_STR.append(iNR_STR)
        INR_END.append(iNR_END)

        # ************************************************************************************* #
        # *                       Plot the aperture photometry                                * #
        # ************************************************************************************* #
        def circle(x, y, r):
            theta = np.linspace(0, 2*np.pi, 100)
            x1 = r*np.cos(theta)+y
            x2 = r*np.sin(theta)+x
            return(x2.tolist(), x1.tolist())

        plot_data = np.ma.masked_array(data, masking)
        figsize = 50
        # im = ax[ang,0].imshow(plot_data - sky_o/gain,vmin=-50,vmax=50,cmap='seismic')
        # xi,yi = circle(xo,yo,Aperture_radius)
        # ax[ang,0].plot(xi,yi,color='y',lw=2)
        # xi,yi = circle(xo,yo,Annulus_radius)
        # ax[ang,0].plot(xi,yi ,color='c',lw=2)
        # xi,yi = circle(xo,yo,Annulus_radius+Dan)
        # ax[ang,0].plot(xi,yi ,color='c',lw=2)
        # ax[ang,0].plot(xo,yo,marker='X',color='c',ms=4)
        # ax[ang,0].set_xlim(xo-figsize,xo+figsize)
        # ax[ang,0].set_ylim(yo-figsize,yo+figsize)
        # ax[ang,0].set_title('Ordinary'+image_i.split('/')[-1],fontsize=14)
        # divider = make_axes_locatable(ax[ang,0])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im,cax=cax)

        # im = ax[ang,1].imshow(plot_data - sky_e/gain,vmin=-50,vmax=50,cmap='seismic')
        # xi,yi = circle(xe, ye,Aperture_radius)
        # ax[ang,1].plot(xi,yi,color='y',lw=2)
        # xi,yi = circle(xe, ye,Annulus_radius)
        # ax[ang,1].plot(xi,yi ,color='c',lw=2)
        # xi,yi = circle(xe, ye,Annulus_radius+Dan)
        # ax[ang,1].plot(xi,yi ,color='c',lw=2)
        # ax[ang,1].plot(xe,ye,marker='X',color='c',ms=4)
        # ax[ang,1].set_xlim(xe-figsize,xe+figsize)
        # ax[ang,1].set_ylim(ye-figsize,ye+figsize)
        # ax[ang,1].set_title('ExtaOrdinary'+image_i.split('/')[-1],fontsize=14)
        # divider = make_axes_locatable(ax[ang,1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im,cax=cax)

        # ************************************************************************************* #
        # *                          Save result of photometry                                * #
        # ************************************************************************************* #
        print(image_i.split('/')[-1])
        Photo_Log = Photo_Log.append({'Object': header['OBJECT'],
                                      'Filename': image_i.split('/')[-1],
                                      'set': int(z/4)+1,
                                      'HWPANG': header['RET-ANG2'],
                                      'TIME': epoch,
                                      'JD': np.mean(JD_mean),
                                      'FWHM_o': aper_seri_o,
                                      'FWHM_e': aper_seri_e,
                                      'Aper [pix]': Aperture_radius,
                                      'EXP [s]': exp,
                                      'Ann': Annulus_radius,
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

    I0 = I[0]
    I45 = I[1]
    I22 = I[2]
    I67 = I[3]

    S0 = S[0]
    S45 = S[1]
    S22 = S[2]
    S67 = S[3]

    Rq = np.sqrt(I0/I45)
    Ru = np.sqrt(I22/I67)

    q = (Rq - 1)/(Rq + 1)
    u = (Ru - 1)/(Ru + 1)

    q_ran = Rq/((Rq + 1)**2)  * np.sqrt(S0 + S45)
    u_ran = Ru/((Ru + 1)**2)  * np.sqrt(S22 + S67)

    q_sys = 0
    u_sys = 0

    q_err = np.sqrt(q_ran**2 + q_sys**2)
    u_err = np.sqrt(u_ran**2 + u_sys**2)

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

    STR0, STR45, STR22, STR67 = INR_STR[0], INR_STR[1], INR_STR[2], INR_STR[3]
    END0, END45, END22, END67 = INR_END[0], INR_END[1], INR_END[2], INR_END[3]

    # averaged value of frame-reader value
    rq = np.deg2rad((STR0 + END0 + STR45 + END45)/4.)   # Instrument star, end value ( 0,0,45,45 ) in rad
    ru = np.deg2rad((STR22 + END22 + STR67 + END67)/4.)  # Instrument star, end value ( 22.5,22.5,67.5,67.5) in rad

    qqq = qq - ((q_inst * np.cos(2*rq)) - (u_inst * np.sin(2*rq)))
    uuu = uu - ((q_inst * np.sin(2*ru)) + (u_inst * np.cos(2*ru)))

    # random error of corrected q,u
    qqq_ran = qq_ran
    uuu_ran = uu_ran

    # the systematic errors
    qqq_sys = np.sqrt( qq_sys**2 + (eq_inst * np.cos(2*rq))**2 +
                       (eu_inst*np.sin(2*rq))**2 )
    uuu_sys = np.sqrt( uu_sys**2 + (eq_inst * np.sin(2*ru))**2 +
                       (eu_inst*np.cos(2*ru))**2 )

    # ====================
    # Transform_CelestialCoord
    # ====================
    theta = np.deg2rad(theta_offset)
    the_err = np.deg2rad(err_theta)

    qqqq = qqq * np.cos(2*theta) + uuu*np.sin(2*theta)
    uuuu = -qqq * np.sin(2*theta) + uuu*np.cos(2*theta)

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
    if P**2 >= P_ran**2:
        print('Random error bias correction is done.')
        P_cor = np.sqrt(P**2 - P_ran**2)
    elif P**2 < P_ran**2 :
        print('Due to P < randome error, random error bias correction is NOT done.')
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
    plt.show()

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
    print('Random error bias correction is done.')
    Pcor = 0
else:
    print('Due to P < random error, P = 0% ')
    Pcor = np.sqrt(P**2 - ran_P**2)

# Converted a polarization degree with respect to the scattering plane
theta = 1/2*np.rad2deg(np.arctan2(u_av, q_av))
psang = np.mean(Pol_Log['PsANG'])
if psang+90 > 180:
    pi = psang-90
else:
    pi = psang + 90


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
