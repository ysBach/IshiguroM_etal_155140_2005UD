# Data Reduction

Here, we provide the code that we used to derive polarimetric results in Ishiguro et al. 2021. The contents are

|Notebook, Script, Directory|Explanation|
|:----------------- |--------------- |
|[``Masking_image.py``](Masking_image.py)|The code for making the "Masking image" of FITS file. The "Masking image" masks the 1) nearby stars, 2)cosmic ray, and if data is MSI's data then, 3) the polarization mask area. |
|``NO_Polarimetric_Analysis.py``|The code to do aperture photometry and derive the stokes parameter from Data taken by MSI at Pirka telescope.|
|``NOT_Polarimetric_Analysis.py``|The code to do aperture photometry and derive the stokes parameter from Data taken by ALFOSC+FAPOL at NOT.|
|``NOT_Subtract.py``|The code to do the subtraction technique which removes the nearby stars and the background's gradient in data taken by NOT. This technique is **used only for ALFOSC's data taken at 2018 Sep 12 and 19**.|
|``result_quplane/``| The directory contains the results of NO and NOT as a qu-plane image.|

The polarimetric data is available in Zenodo. 


## Requirement
Before running the script, the following packages must be installed. 

1. [astropy](https://www.astropy.org/) 
2. [Astro-SCRAPPY](https://github.com/astropy/astroscrappy) 
3. [Source Extraction and Photometry](https://sep.readthedocs.io/en/v1.1.x/index.html) 



## How to use
Let me take the MSI's data taken at 20180925 as an example.
    
1. **First, let's check the directory. Your directory must contain the following:**
    * <u>Polarimetric pre-processed data (FITS format)</u>.
     All sets should be consisted of 4 images (taken at HWP=0, 22.5, 45, 67.5 deg).
      If even one set does not have 4 images (e.g., set having images taken at HWP = 0, 45, 67.5 deg), an error will occur. 
      In other words, the number of data in the directory must be a multiple of 4. 
    
    * ``*.mag.1`` file (file made with IRAF's ``DAOPHOT`` task) containing the center information of the target.
    
    * (If you are processing MSI data) **A directory named '``Flat/``'** where the dome flat image is stored.
    * Data, ``*.mag.1`` file and the dome flat image are provied in the Zenodo Repository.
    
2. **Making the masking image by with ``Masking_image.py``.**
   
    * Fill in the <i>INTPUR VALUE</i> 
    
    * For example, 
    
      ```python
      #*******************************#
      #*        INPUT VALUE          *#
      #*******************************#
      INSTRUMENT = 'MSI'  # Choose which data you use. one of 'MSI' or 'NOT'
      Observatory = 'q33' # Nayoro observatory: 'q33', NOT: 'Z23'
      OBJECT = 155140  # 2005 UD
      Limiting_mag = 18  # Masking the star brighter than the given magnitude
      Pill_masking_with = 30  # [pix] #The Width of pill box masking the nearby stars
      path = os.path.join('/home/judy/20180927')  # Path of directory where images to be masked are saved
      IMAGE_list = glob.glob(os.path.join(path,'msi.fits'))  # Bring all FITS preprocessed image to be masked in the directory
      ```
    
    * Run the script.
    
3. **Deriving the stokes parameter by using ``NOT_Polarimetric_Analysis.py`` or ``NO_Polarimetric_Analysis.py`` for ALFOSC data or MSI data respectively.**
   
    * Fill in the <i>INPUT VALUE</i>.
    * For example,
        ```python
        #*******************************#
        #*        INPUT VALUE          *#
        #*******************************#
        Observatory = 'q33' # Observatory code of Pirks
        OBJECT = 155140  # 2005 UD
        path = os.path.join('/home/judy/20180927')           # Path of directory where images to be masked are saved
        IMAGE_list = glob.glob(os.path.join(path,'*.fits'))  # Bring all FITS preprocessed image in the directory
        #####################################
        # Photometry Parameter
        #####################################
        Aperture_scale = 1.7  # Aperture radius = Aperture_scale * FWHM
        ANN_sacle = 4         # Annulus radius = ANN_sacle * FWHM
        Dan = 20              # [pix] #Dannulus size 
        ####################################
        # Calibration Parameter (Please see Ishiguro et al. 2021)
        ####################################
        eff = 0.9913         # Polarimetric efficiency of instrument
        eff_err = 0.0001     # Error of polarimetric efficiency of instrument
        q_inst = 0.00791     # Instrumental polarization, q{inst}
        u_inst = 0.00339     # Instrumental polarization, u{inst}
        eq_inst = 0.00025    # Error of instrumental polarization, error of q{inst}
        eu_inst = 0.00020    # Error of instrumental polarization, error of u{inst}
        theta_offset = 3.66  # Offset of polarization angle 
        err_theta = 0.17     # Error of offset
        
    * Run the script.
    
4. **If you want to apply the subtraction technique, run ``NOT_Subtraction.py`` from the beginning and then do step 3.** (Do not make the masking image in this case.)

    


## Out-put
When you run the script you will get two files (``result_Photo_XXXX.csv`` and ``result_Pol_XXXX.csv`` where XXXX is the observation date).

Each csv file contains the photometric result of each image and the polarimetric result of each set.
The header information is below.
    

### ``result_Photo_<ObservationDate>.csv``
|Header Keyword   |Unit|Explanation|
| ----------------- |---| --------------- |
|Object Filename| ... |Name of FITS file|
|set| ... | Set number|
|TIME| ... | Mid-point of exposure in UT|
|JD| ... |Mid-point of exposure in JD|
|HWPANG|deg|Retarder Plate Angle |
|FWHM_o|pixel|FWHM of the ordinary component|
|FWHM_e|pixel|FWHM of the extra-ordinary component|
|EXP [s]|second|Exposure time|
|Aper [pix]|pixel|Aperture radius|
|Ann|pixel|Inner circle radius of the annulus|
|Ann_out|pixel|Outher circle radius of the annulus|
|Flux_o|e-|Sum of flux within the aperture of the ordinary component|
|eFLux_o|e-|Error of Flux_o|
|Flux_e|e-|Sum of flux within the aperture of the extra-ordinary component|
|eFLux_e|e-|Error of Flux_e|
|Sky_o|e-|Background value around the ordinary component|
|eSky_o|e-|Error of Sky_o|
|Sky_e|e-|Background value around the extra-ordinary component|
|eSky_e|e-|Error of Sky_e|
|SNR_o|...|SNR of the ordinary component|
|SNR_e|...|SNR of the extra-ordinary component|

​    

### ``result_Pol_<ObservationDate>.csv``
|Header Keyword   |Unit|Explanation|    
| :----------------- |---| --------------- |
|filename|...|Name of FITS file|
|JD|...|Mid-point of set in JD|
|alpha [deg]|deg|Average phase angle of set|
|PsANG [deg]|deg|Average position angle of the scattering plane of set|
|q|...|q of the Stokes parameters|
|u|...|u of the Stokes parameters|
|ran_q|...|Random error of q|
|ran_u|...|Random error of u|
|sys_q|...|Systemic error of q|
|sys_u|...|Systemic error of u|
|P|...|Linear polarization degree|
|eP|...|Error of P|
|Pr|...|Polarisation degree referring to the scattering plane|
|theta|deg|Positino angle|
|theta_r|deg|Position angle referring to the scattering plane|
|eTheta|deg|Error of theta|
|Aper_radius [pix]|pixel|Aperture radius|


​    
## Contact

Created by Jooyeon Geem. - If you have any questions, please feel free to contact me (geem@astro.snu.ac.kr) !

The data reduction pipeline of MSI will continue to be developed in [@Geemjy](https://github.com/Geemjy) in the future.

