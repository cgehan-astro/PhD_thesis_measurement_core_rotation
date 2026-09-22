### Organisation of the repository

There are 4 programs in this repository.


### 1. Synthetic_echelle_diagram.py

This program produces as an output the plot synthetic_echelle_dnurot_dnumag.pdf.


### 2. Stretching_spectrum.py

This program takes as an input the files ./Stars/parameters_KIC_*.txt and produces as an output the files ./Stars/stretched_spectrum_KIC_*.txt.


### 3. Core_rotation_measurement.py

This program takes as inputs the files ./Stars/stretched_spectrum_KIC_*.txt and ./Stars/parameters_KIC_*.txt. This program produces as outputs the file Results.txt, three plots saved in ./Figures_rotation_intermediate/ and the plot Echelle_diagram_KIC_*.pdf.


### 4. Results.py

This program takes as inputs the files Core_rotation_measurements.txt and APOKASC.txt. This program produces as outputs the plots M_R.pdf and Evolution_core_rotation_colored_with_mass.pdf.


### Installation: with anaconda

git clone https://github.com/cgehan-astro/PhD_thesis_measurement_core_rotation.git
cd PhD_thesis_measurement_core_rotation
conda env create -f environment.yml
