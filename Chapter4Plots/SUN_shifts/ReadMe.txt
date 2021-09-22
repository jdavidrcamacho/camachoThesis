J/A+A/635/A13  HD41248 Radial velocities and activity indicators  (Faria+, 2020)
================================================================================
Decoding the radial velocity variations of HD 41248 with ESPRESSO.
    Faria J.P., Adibekyan V., Amazo-Gomez E.M., Barros S.C.C., Camacho J.D.,
    Demangeon O., Figueira P., Mortier A., Oshagh M., Pepe F., Santos N.C.,
    Gomes da Silva J., Costa Silva A.R., Sousa S.G., Ulmer-Moll S., Viana P.T.P.
    <Astron. Astrophys. 635, A13 (2020)>
    =2020A&A...635A..13F        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Stars, double and multiple ; Exoplanets ; Radial velocities
Keywords: techniques: radial velocities - methods: data analysis -
          planetary systems - stars: individual: HD 41248

Abstract:
    Twenty-four years after the first exoplanet discoveries, the
    radial-velocity (RV) method is still one of the most productive
    techniques to detect and confirm exoplanets. But stellar magnetic
    activity can induce RV variations large enough to make it difficult to
    disentangle planet signals from the stellar noise. In this context,
    HD 41248 is an interesting planet-host candidate, with RV observations
    plagued by activity-induced signals. We report on ESPRESSO
    observations of HD 41248 and analyse them together with previous
    observations from HARPS, with the goal of evaluating the presence of
    orbiting planets. Using different noise models within a general
    Bayesian framework designed for planet detection in RV data, we test
    the significance of the various signals present in the HD 41248 data
    set. We use Gaussian processes as well as a first-order moving average
    component to try to correct for activity-induced signals. At the same
    time, we analyse photometry from the TESS mission, searching for
    transits and rotational modulation in the lightcurve. The number of
    significantly detected Keplerian signals depends on the noise model
    employed, ranging from 0 with the Gaussian process model to 3 with a
    white noise model. We find that the Gaussian process alone can explain
    the RV data and allows for the stellar rotation period and active
    region evolution timescale to be constrained. The rotation period
    estimated from the RVs agrees with the value determined from the TESS
    lightcurve. Based on the currently available data, we conclude that
    the RV variations of HD 41248 can be explained by stellar activity
    (using the Gaussian process model) in line with the evidence from
    activity indicators and the TESS photometry.

Description:
    Full set of radial velocities and activity indicators from both the
    HARPS and ESPRESSO observations of HD41248

Objects:
    --------------------------------------------------
       RA   (2000)   DE        Designation(s)
    --------------------------------------------------
    06 00 32.78  -56 09 42.6   HD 41248 = HIP 28460
    --------------------------------------------------
File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat        78      250   Radial velocities and activity indicators
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 12  F12.6 d       BJD       Modified barycentric Julian day (BJD-2400000)
  14- 20  F7.5  km/s    BRV       Barycentric radial velocity
  22- 28  F7.5  km/s  e_BRV       BRV error
  30- 36  F7.5  km/s    FWHM      CCF FWHM
  38- 45  F8.5  m/s     BIS       CCF bisector span
  47- 53  F7.5  ---     ICaII     Activity index based on the Ca II H & K lines
  55- 61  F7.5  ---     IHa       Activity index based on the Halpha line
  63- 69  F7.5  ---     INaI      Activity index based on the Na I lines
  71- 78  A8    ---     Inst      Instrument (1)
--------------------------------------------------------------------------------
Note (1): Instrument column is either "HARPS" or "ESPRESSO"
--------------------------------------------------------------------------------

Acknowledgements:
    Joao Pedro Faria, joao.faria(at)astro.up.pt

================================================================================
(End)   Joao Pedro Faria [IA, Portugal], Patricia Vannier [CDS]     22-Nov-2019
