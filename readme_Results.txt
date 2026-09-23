### Project: Results.py

I completed this project during my PhD thesis (2015 - 2018).


### Project overview

This project aims at providing a physical interpretation of the measurements of the core rotation rate provided by the pipeline presented in Core_rotation_measurement.py


### Dataset

The dataset is composed of the physical parameters contained in the file Core_rotation_measurements.txt, as well as an additional physical parameter from the file APOKASC.txt: the effective temperature, i.e. the temperature at the surface of stars.


### Results

Two plots are saved, representing two examples of what we can learn from the data: M_R.pdf and Evolution_core_rotation_colored_with_mass.pdf.

1. The plot M_R.pdf represents the mass of red giant stars as a function of their radius, both estimated based on scaling relations using physical parameters. It highlights that there is a significant correlation between the stellar mass and radius associated to a Pearson correlation coefficient of 0.495 and a p-value of 1.889 * 10**(-53).

2. The plot Evolution_core_rotation_colored_with_mass.pdf represents the core rotation rate in frequency as a function of a proxy for the degree of evolution of stars estimated based on a scaling relation using physical parameters obtained with asteroseismology. The color code represents the stellar mass and the black dashed lines indicate a fit with a power-of-ten function.


### Key findings

 1. The plot M_R.pdf indicates that although red giant stars expand as they evolve, their radius is not an appropriate proxy for their degree of evolution due to its strong dependence on their mass.

2. The plot Evolution_core_rotation_colored_with_mass.pdf emphasizes two key results that are widely used by theorists, simulators, and modelers because they are central to identifying the physical mechanisms that govern stars.
- The first key result is that the core rotation rate remains constant for red giant stars along their evolution, with an exponent compatible with 0. However, we expect on the contrary that the core rotation rate should accelerate as a result of angular momentum conservation, because the core is contracting. This result indicates the existence of a significant transport of angular momentum inside red giant stars, which no physical mechanism has yet fully explained.
 - The second key result is that the core rotation values and evolution are independent of the mass of red giant stars. However, this is not expected because more massive stars evolve faster, on a shorter timescale, which gives less time to extract angular momentum from their core. This result indicates that angular momentum is extracted from the core of red giant stars faster and therefore more efficiently when the stellar mass increases, which provides an additional clue to indetify the physical mechanism(s) at work.


### Installation: with anaconda

git clone https://github.com/cgehan-astro/PhD_thesis_measurement_core_rotation.git
cd PhD_thesis_measurement_core_rotation
conda env create -f environment.yml
