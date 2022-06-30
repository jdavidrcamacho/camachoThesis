# Thesis

## Advanced Statistical Data Analysis Methods for the Detection of Other Earths

The discovery of the first extra-solar planet was a breakthrough that revo-
lutionized our understanding of the Universe. The increasing number of planets
discovered combined with the growing instrumental precision allows us to be
closer to the new milestone of finding an Earth-like planet around a Sun-like star.
As we approach this discovery, new challenges arise and with it the need for ever
more powerful data analysis methods.

For this thesis, we present a new Gaussian process framework to improve
radial velocity measurements analysis. This new framework, known as Gaussian
process regression network, combines the stellar activity information on auxiliary
datasets to improve the identification and removal of these signals from the respec-
tive radial velocity time series. The novel concept of a Gaussian process regression
network lies in its non-stationary setup, ideal for modelling stellar activity signals.
We describe all the mathematical foundations necessary to implement a
Gaussian process regression network and understand its differences from other
frameworks in use. To facilitate its use, we created a python package that we
employed to analyze radial velocity observations of the Sun. The results allowed
us to comprehend the complexity and constraints our work possesses. It showed
that the Gaussian process regression network, in its current form, has limitations
not exhibited on a traditional Gaussian process regression. Notwithstanding,
it allowed to identify several steps required to improve the capabilities of the
regression network.

We also applied this new framework to four stars possessing different stellar
activity levels. Done for the EXPRES Stellar-Signals Project, it allowed us to
determine that the Gaussian process regression network shown to be sensitive to
the stellar activity level presenting better results on more active stars. These results
were also crucial to conclude that, even considering its limitations, the regression
network obtains results similar to other time series modelling methods.


### Cite

If you use this thesis in your work, please cite the following publication
```bibtex
@PHDTHESIS{Camacho2022_phd,
       author = {{Camacho}, Jo{\~a}o David Ribeiro},
       title = "{Advanced Statistical Data Analysis Methods for the Detection of Other Earths}",
       school = {University of Porto, Portugal},
       year = 2022,
       month = jun,
}
```
