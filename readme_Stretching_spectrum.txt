### Project: Stretching_spectrum.py

I completed this project during my PhD thesis (2015 - 2018).


### Project overview

This project aims at stretching power spectra in order to force a regular spacing of the peaks making much easier the measurement of the physical parameter we are interested in: the rate at which the core of stars rotates on itself, i.e. the core rotation of stars. This is made possible by the propagation of internal waves inside stars that generates oscillation modes, allowing us to probe the interior of stars: this field is called asteroseismology, i.e. stellar seismology that works on principle similar as for Earth's seismology. In the case of red giant stars, which are evolved stars representing the future evolution stage of our Sun, some oscillation modes probe the deepest regions of the stars, i.e. their core, making possible to measure their core rotation rate.


### Dataset

The dataset is composed of power spectra obtained beforehand by taking the Fourier transform of the light curves of stars, i.e. the variation of their flux over time; indeed, the oscillation modes generate extremely faint and periodic variations of the flux of stars, making possible to identify the oscillation modes by their frequency in the power spectrum. The dataset is composed of power spectra for about 2000 red giant stars, i.e. one power spectrum per star. Here, only one power spectrum is provided as an example: ./Stars/parameters_KIC_*.txt.


### Methodology

The power spectra are stretched in order to force a regular spacing of the modes, by integrating a physical differential equation that encodes the nature of the oscillation modes. Depending on their frequency, some modes are are almost regularly spaced in frequency, i.e. pressure-dominated modes, while some modes are almost regularly spaced in period, i.e. gravity-dominated modes. The pressure-dominated modes are the modes that are most stretched, i.e. which spacing is most increased, compared to the gravity-dominated modes that are only sligthly stretched.


### Results

The resulting power spectra in stretched period are saved in the files ./Stars/stretched_spectrum_KIC_*.txt and are then exploited by the program Core_rotation_measurement.py.


### Installation: with anaconda

git clone https://github.com/cgehan-astro/PhD_thesis_measurement_core_rotation.git
cd PhD_thesis_measurement_core_rotation
conda env create -f environment.yml
