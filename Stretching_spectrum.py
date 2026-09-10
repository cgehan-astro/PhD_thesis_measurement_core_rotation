### Create the appropriate environment with anaconda: conda env create -f environment.yml
### python=3.9.25, ipython=8.15


### This program aims at stretching power spectra in order to force a regular spacing of the peaks using a physically-motivated basis, making much easier the measurement of the physical parameter we are interested in

import numpy as np
import pylab as plt
import csv
import os
import glob



### Function required to stretch the spectra that depends on frequency according to physical laws

def calculation_zeta(radial_order, p_freq, Dnu_np, n, Dpi1):
    list_zeta = []
    list_nu = []
    for j in range(1, radial_order.size): 	# loop required to iteratively build parts of the function over different frequency ranges
    
    ## Definition of the frequency range used to compute the function
   
        x = (p_freq[j - 1] + p_freq[j]) / 2 			
        nu_0 = x + Dnu_np[j] * np.array(range(n)) / n
	
	
    ## Computing the function itself for a given frequency range
	
        X = 10.**(-6) * nu_0 ** 2 * Dpi1 / (q * Dnu_np[j])	
        Y = np.pi * (nu_0 - p_freq[j]) / Dnu_np[j]
        Z = 1. / q ** 2 * np.sin(Y) ** 2 + np.cos(Y) **2
        zeta_0 = (1 + X * 1. / Z) ** (-1)
	
    ##	Loop to append and iteratively construct the function over frequency from the computation for each frequency range 
	
        for k in range(zeta_0.size):
            list_zeta.append(zeta_0[k])
            list_nu.append(nu_0[k])
    zeta = np.array(list_zeta)
    nu = np.array(list_nu)
    return nu, zeta



### Function that stretches the spectra, turning an evenly-spaced frequency frequency spectrum into a regularly-spaced period spectrum according to physical laws

def calculation_tau(freq, zeta):
	list_tau = []
	for j in range(freq.size):
	    crit_int = np.where(freq <= freq[j])[0]
	    tau_0 = 10.**6 / (crit_int.size)  * (freq[j] - min(freq)) * np.sum(1./(zeta[crit_int] * freq[crit_int]**2))	# integration
	    list_tau.append(tau_0)
	tau = np.array(list_tau)
	return tau



### Reading the physical parameters needed beforehand from the input files

path = './Stars/parameters*.txt'

files = sorted(glob.glob(path))
files_number = np.size(files, axis=0)



### Beginning of the main program

## Reading the physical parameters needed beforehand

for i in range(files_number):						# loop on the different files
    mat_freq_star = np.loadtxt(files[i], skiprows=14)
    size_mat = np.size(mat_freq_star,axis=0)

    freq_0 = mat_freq_star[:,0]						# frequency column
    spec_dens = mat_freq_star[:,2]					# power spectral density column

    lignes_number = np.size(mat_freq_star, axis=0)
    col_nu = np.size(mat_freq_star, axis=1)

    desired=[11,12,13]
    with open(files[i], 'r') as fin:
        reader=csv.reader(fin)
        result=[[file_ind for file_ind in row] for number,row in enumerate(reader) if number in desired]
        result = np.array(result)

	## Separating the array in three sub-arrays because the number of columns differs for every line

        result_0 = result[0,0].replace('\t0.', '')	# getting rid of brackets
        result_0 = result_0.split(" ")			# splitting the array according to blank spaces

        result_1 = result[1,0].replace('\t0.', '')
        result_1 = result_1.split(" ")

        result_2 = result[2,0].replace('\t0.', '')
        result_2 = result_2.split(" ")

        mat_0_list = []
        for j in range(np.size(result_0, axis=0)):	# obtaining an array of values by getting rid of the blank spaces 
            if list(result_0[j]):
                mat_0_list.append(float(result_0[j]))

        mat_1_list = []
        for j in range(np.size(result_1, axis=0)):
            if list(result_1[j]):
                mat_1_list.append(float(result_1[j]))

        mat_2_list = []
        for j in range(np.size(result_2, axis=0)):
            if list(result_2[j]):
                mat_2_list.append(float(result_2[j]))

        mat_0 = np.array(mat_0_list)
        mat_1 = np.array(mat_1_list)
        mat_2 = np.array(mat_2_list)

    Dnu = mat_0[0]
    nu_max = mat_0[1]
    epsilon_p = mat_0[2]
    d01 = mat_0[3]
    Dpi1 = mat_1[0]
    q = mat_1[1]
    cfon = mat_2[0]
    expo = mat_2[1]
    
    
    ## Definition of other needed parameters based on scaling relations

    alpha = 0.015 * Dnu ** (-0.32)
    n_max = (nu_max / Dnu) - epsilon_p
    

    ## Definition of the frequency interval other which computing the function enabling stretching the spectra

    d_n = 6
    n_p = np.linspace(int(n_max - d_n), int(n_max + d_n), int(n_max + d_n) + 1 - int(n_max - d_n)).astype(int)
       
    Dnu_np = Dnu * (1 + alpha * (n_p - n_max))					# correcting the input theoretical parameter Dnu to account for a physical effect that depends on frequency 
    nu_p = (n_p + epsilon_p + d01 + (alpha / 2) * (n_p - n_max) ** 2) * Dnu	# defintion of the theoretical frequencies nu_p used as a basis to compute the function enabling stretching the spectra based on the input physical parameters


    ## Computation of the function enabling stretching the spectra

    N_dens = 100
    n = int(10 ** 6 * Dnu / ((nu_max) ** 2 * Dpi1) * N_dens)  # number of points used to compute the function with a high enough resolution
    freq = mat_freq_star[:,0]
    spec_dens = mat_freq_star[:,2]

    nu, zeta = calculation_zeta(n_p, nu_p, Dnu_np, n, Dpi1)
    zeta_all_freq = np.interp(freq, nu, zeta)		      # interpolating the function at the actual frequency values that are present in the spectrum


    ## Stretching the spectrum

    tau = calculation_tau(freq, zeta_all_freq)
    

    ## Saving the results in an output file

    star = files[i]
    name = star.replace('./Stars/parameters_KIC_', '')
    name = name.replace('.txt', '')
    output_file_tau = './Stars/stretched_spectrum_KIC_' + name + '.txt'
    if os.path.exists(output_file_tau) == False:
        fichier_output_file_tau = open(output_file_tau, "w")
        fichier_output_file_tau.write('nu (muHz)   tau (s)   PSD' + '\n')
        for j in range(tau.size):
            fichier_output_file_tau.write(str(mat_freq_star[j,0]) + ' ' + str(tau[j]) + ' ' + str(mat_freq_star[j,2]) + '\n')
        fichier_output_file_tau.close()

