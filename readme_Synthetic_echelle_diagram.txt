### Project: Synthetic_echelle_diagram.py

I completed this project during my PhD thesis (2015 - 2018).


### Project overview

This project aims at building a synthetic power spectrum in order to visualize in an easy way the effect of two physical parameters we are interested in on the frequency of oscillation modes: rotation and magnetic fields. The power spectrum is stretched in order to force a regular spacing of the modes; this spacing corresponds to the gravity-mode period spacing Dpi1. In order to visualize the effects of rotation and magnetic fields, we rely on an échelle diagram, which represents the frequency of oscillation modes as a function of their stretched period modulo Dpi1, i.e. folded with Dpi1. Rotation and magnetic fields perturbs the oscillation modes, resulting in a departure from the regular spacing Dpi1 of the modes.


### Methodology

The synthetic power spectrum is built from input physical parameters using physical equations and scaling relations. It is then stretched in order to force a regular spacing of the modes, by integrating a physical differential equation that encodes the nature of the oscillation modes. Depending on their frequency, some modes are are almost regularly spaced in frequency, i.e. pressure-dominated modes, while some modes are almost regularly spaced in period, i.e. gravity-dominated modes. The pressure-dominated modes are the modes that are most stretched, i.e. which spacing is most increased, compared to the gravity-dominated modes that are only sligthly stretched.


### Results

The resulting synthetic échelle diagram is saved as synthetic_echelle_dnurot_dnumag.pdf. In the hypothetical absence of rotation and magnetic fields, all the modes would align along the vertical light blue ridge. Rotation modifies this scheme by splitting the oscillation modes into two more components, represented in light red and light green. Magnetic fields add an additional perturbation to all the three components, resulting in the dark blue, dark red and dark green components. Such échelle diagrams are used in the program Core_rotation_measurement.py as visual tools to identify the number of visible rotational components, a pre-requisite to correctly measure the core rotation rate of red giant stars.


### Installation: with anaconda

git clone https://github.com/cgehan-astro/PhD_thesis_measurement_core_rotation.git
cd PhD_thesis_measurement_core_rotation
conda env create -f environment.yml
