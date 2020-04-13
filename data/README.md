## Polarimetric data

| column        | meaning                                                      | dtype            |
| ------------- | ------------------------------------------------------------ | ---------------- |
| id            | The permanent number of the asteroid.                        | int              |
| name          | The name of the asteroid, if exists.                         | str              |
| altname       | The provisional designation; only the most widely used one (e.g., main name of SBDB) if there are multiple. | str              |
| date          | UT date of the mean observation time.                        | str (YYYY-MM-DD) |
| middletime    | UT time of the mean observation (I hope references did lighttime correction: I have **not** checked this). Sometimes this middle time is not easy to calculate, so I just took mean of start/end of the set of observation as far as the original publication gave. | str (HH:MM:SS)   |
| jd            | Julian date of ``<date>T<middletime>``                       | float            |
| alpha         | Phase angle in degrees (˚).                                  | float            |
| value         | The proper polarization degree in % ($\frac{I_\perp - I_\parallel}{I_\perp + I_\parallel}$). | float            |
| value_err     | The uncertainty of ``value``. Some references either (1) do not provide this or (2) it is standard deviation not the standard error from CLT. For these cases, (1) I adopted the error for total polarization degree without any correction and added a comment (2) used it as is and added a comment. | float            |
| reference     | ADS format reference.                                        | str              |
| obsmode       | mode of the observation. Always pol... so far.               | str              |
| filter        | Filter name following the original publication (``Rc``, ``RC``, ``0.65``, etc) | str              |
| weighted_mean | Whether it is the value from weighted mean (i.e., least squar fit of a constant function) | int (0/1)        |
| N             | Number of frames used, not the number of _sets_ used. In some cases, it is not clear whether their ``n`` is number of sets or frames. I just naively put their ``n`` value without much consideration. | int              |
| exptime       | The exposure time in seconds.                                | float            |
| comment       |                                                              | str              |
| airmass       |                                                              | float            |

