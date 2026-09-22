### Create the appropriate environment with anaconda: conda env create -f environment.yml
### python=3.9.25, ipython=8.15

### This program aims at measuring a physical parameter from a power spectrum in stretched period, i.e. a frequency power spectra stretched in order to force a regular spacing of the peaks using a
### physically-motivated basis, making much easier the measurement of the physical parameter we are interested in

import numpy as np
import pylab as plt
import glob
import re
import csv
import os



### Routine calculating pure pressure mode frequencies with the universal pattern

def function_modes_frequencies(d_n, l, n_max, epsilon_p, d_0_l, alpha, Dnu):
	b = []
	n_p_low = int(n_max - d_n)
	n_p_up = int(n_max + d_n)
	for n_p in range(n_p_low, n_p_up + 1):
		freq_p = (n_p + epsilon_p + d_0_l + (alpha/2) * (n_p - n_max)**2) * Dnu
		b.append(freq_p)
	nu_p = np.array(b)
	return nu_p



### Routine finding pressure and gravity dominated modes in the spectrum

def function_find_pressure_gravity_modes(lignes_number, col_nu, mat, lignes_nu_p, nu_p, epsilon, B):
	c = []
	d = []
	e = []
	gravity_modes = np.zeros((lignes_number, col_nu))
	gravity_modes[:] = mat[:]
	for j in range(lignes_nu_p):
		diff = mat[:,0] - nu_p[j]
		p_modes_crit = np.where(abs(diff) <= epsilon)[0]
		if list(p_modes_crit) != []:
			p_modes = mat[p_modes_crit,:]
			B_p_modes_ini = B[p_modes_crit]
			size_p_modes = np.size(p_modes, axis=0)
			for k in range(size_p_modes):
				c.append(p_modes[k,:])
				d.append(B_p_modes_ini[k])
				e.append(p_modes_crit[k])
	pressure_modes = np.array(c)
	B_p_modes = np.array(d)
	index_p_modes = np.array(e)
	gravity_modes = np.delete(gravity_modes, index_p_modes, axis=0)
	B_g_modes = np.delete(B, index_p_modes)
	return pressure_modes, B_p_modes, gravity_modes, B_g_modes



### Routine selecting peaks above x_times_back times the background : p and g-m modes

def function_peaks_above_noise(lignes_pressure_modes_ini, pressure_modes_ini, B_p_modes, lignes_gravity_modes_ini, gravity_modes_ini, B_g_modes, x_times_back):
	a = []
	for j in range(lignes_pressure_modes_ini): 
		if pressure_modes_ini[j,2] > x_times_back * B_p_modes[j]:
			a.append(pressure_modes_ini[j,:])
	good_values_ini_p = np.array(a)
	b = []
	for j in range(lignes_gravity_modes_ini): 
		if gravity_modes_ini[j,2] > x_times_back * B_g_modes[j]:
			b.append(gravity_modes_ini[j,:])
	good_values_ini_g = np.array(b)
	return good_values_ini_p, good_values_ini_g



### Routine weighting the power spectral densities for plotting considerations

def function_weighting_PSD(good_values_ini_p, good_values_ini_g, col_nu):
	lignes_pressure_modes = np.size(good_values_ini_p, axis=0)
	lignes_gravity_modes = np.size(good_values_ini_g, axis=0)
	pressure_modes = np.zeros((lignes_pressure_modes, col_nu))
	gravity_modes = np.zeros((lignes_gravity_modes, col_nu))
	pressure_modes[:] = good_values_ini_p[:]
	gravity_modes[:] = good_values_ini_g[:]
	pressure_modes[:,2] = good_values_ini_p[:,2] / d
	gravity_modes[:,2] = good_values_ini_g[:,2] / d
	return lignes_pressure_modes, lignes_gravity_modes, pressure_modes, gravity_modes



### Routine calculating the centroid of gravity dominated mixed modes aligned horizontally that have a too low frequency dispersion and replaces these frequencies by the centroid values

def function_centroid_points(centroid_mat, Dpi1, centroid_crit, absc_crit, gravity_par):
	centroid_points = np.array(centroid_mat)
	size_centroid_points = np.size(centroid_points, axis=0)
	tau_mod_Dpi1_centroid = centroid_points[:,1] % Dpi1
	centroid_points[:,3] = tau_mod_Dpi1_centroid
	centroid_points = np.c_[centroid_points, np.zeros((size_centroid_points))]
	freq_diff = np.diff(centroid_points[:,0])
	redundant_freq_crit = np.where(freq_diff != 0.)
	if max(redundant_freq_crit[0]) == size_centroid_points - 2:
		redundant_freq_list = list(redundant_freq_crit[0])
		redundant_freq_list.append(max(redundant_freq_crit[0]+1))
	redundant_freq_crit = np.array(redundant_freq_list)
	centroid_points = centroid_points[redundant_freq_crit,:]
	size_centroid_points = np.size(centroid_points, axis=0)
	diff_centroid_points = np.diff(centroid_points[:,0])
	diff_absc_points = np.diff(centroid_points[:,3])
	count = 0.
	for j in range(size_centroid_points-1):							# loop finding the different groups of peaks to center
		centroid_points[j+1,4] = count
		if abs(diff_centroid_points[j]) > centroid_crit and abs(diff_absc_points[j]) > absc_crit:
			count = count + 1
			centroid_points[j+1,4] = count
		if abs(diff_centroid_points[j]) > centroid_crit and abs(diff_absc_points[j]) < absc_crit:
			count = count + 1
			centroid_points[j+1,4] = count
		if abs(diff_centroid_points[j]) < centroid_crit and abs(diff_absc_points[j]) > absc_crit:
			count = count + 1
			centroid_points[j+1,4] = count
	centroid_select = np.array(list(set(centroid_points[:,4])))
	size_centroid_select = np.size(centroid_select)
	b = []
	for j in range(size_centroid_points):							# loop deleting values to center in the original array
		gravity_par_crit_0 = np.where(gravity_par[:,0] == centroid_points[j,0])
		size_gravity_par_crit = np.size(gravity_par_crit_0, axis=0)
		for k in range(size_gravity_par_crit):
			b.append(gravity_par_crit_0[k])
	gravity_par_crit = np.array(b)
	gravity_par_centroid = np.delete(gravity_par, gravity_par_crit, axis=0)
	for j in range(size_centroid_select):							# loop taking the centroid of the values
		centroid_crit = np.where(centroid_points[:,4] == centroid_select[j])
		centroid = centroid_points[centroid_crit,:][0]
		size_centroid = np.size(centroid, axis=0)
		m = []
		n = []
		for k in range(size_centroid):
			tau_centroid_ini = centroid[k,1] * centroid[k,2]
			freq_centroid_ini = centroid[k,0] * centroid[k,2]
			m.append(tau_centroid_ini)
			n.append(freq_centroid_ini)
		tau_centroid_0 = np.array(m)
		freq_centroid_0 = np.array(n)
		tau_centroid = np.sum(tau_centroid_0) / np.sum(centroid[:,2])
		freq_centroid = np.sum(freq_centroid_0) / np.sum(centroid[:,2])
		gravity_par_centroid = np.r_[gravity_par_centroid, [np.zeros(np.size(gravity_par_centroid, axis=1))]]		# adding the centroid values to the original array
		size_gravity_par_centroid = np.size(gravity_par_centroid, axis=0)
		gravity_par_centroid[size_gravity_par_centroid-1,0] = freq_centroid
		gravity_par_centroid[size_gravity_par_centroid-1,1] = tau_centroid
		gravity_par_centroid[size_gravity_par_centroid-1,2] = np.mean(centroid[:,2])
	sort_gravity_par_centroid = np.argsort(gravity_par_centroid[:,0])
	gravity_par_centroid = gravity_par_centroid[sort_gravity_par_centroid,:]
	return gravity_par_centroid



### Routine grouping the peaks of interest to identify the oscillation modes: one single oscillation mode can result in more than one peak due to the stochastic nature of oscillations and the non-zero frequency resolution

def function_centroid_peaks(gravity_modes, lignes_gravity_modes, centroid_crit, absc_crit, Dpi1):
	freq_ini = gravity_modes[:,0]			# frequencies for gravity-dominated mixed modes
	tau_ini = gravity_modes[:,1]			# stretched periods for gravity-dominated mixed modes
	spec_dens_ini = gravity_modes[:,2]		# spectral densities for gravity-dominated mixed modes
	tau_mod_Dpi1_ini = tau_ini % Dpi1		# stretched periods modulo Dpi1 for gravity-dominated mixed modes
	freq_p = pressure_modes[:,0]			# frequencies for pressure-dominated mixed modes
	tau_p = pressure_modes[:,1]			# stretched periods for pressure-dominated mixed modes
	spec_dens_p = pressure_modes[:,2]		# spectral densities for pressure-dominated mixed modes
	tau_mod_Dpi1_p = tau_p % Dpi1			# stretched periods modulo Dpi1 for pressure-dominated mixed modes
	sort_freq_crit = np.argsort(gravity_modes[:,0])
	gravity_par = gravity_modes[sort_freq_crit,:]			# sorting the frequencies values
	centroid_mat = []
	for j in range(lignes_gravity_modes-1):				# looking for the existence of peaks too close in frequency
		freq_diff = gravity_par[j+1,0] - gravity_par[j,0]
		if freq_diff < centroid_crit:
			centroid_mat.append(gravity_par[j,:])
			centroid_mat.append(gravity_par[j+1,:])
	if centroid_mat:						# case where there are some peak too close in frequency
		centroid_mat = np.c_[centroid_mat, np.zeros(np.size(centroid_mat, axis=0))]
		gravity_par_centroid = function_centroid_points(centroid_mat, Dpi1, centroid_crit, absc_crit, gravity_par) 	# finding the centroid of the gravity-dominated mixed modes
		lignes_gravity_modes = np.size(gravity_par_centroid, axis=0)
		freq = gravity_par_centroid[:,0]			# frequencies for gravity-dominated mixed modes
		tau = gravity_par_centroid[:,1]				# stretched periods for gravity-dominated mixed modes
		spec_dens = gravity_par_centroid[:,2]			# spectral densities for gravity-dominated mixed modes
		tau_mod_Dpi1 = tau % Dpi1				# stretched periods modulo Dpi1 for gravity-dominated mixed modes
	else:								# case where there are not any peaks too close in frequency
		freq = gravity_modes[:,0]				# frequencies for gravity-dominated mixed modes
		tau = gravity_modes[:,1]				# stretched periods for gravity-dominated mixed modes
		spec_dens = gravity_modes[:,2]				# spectral densities for gravity-dominated mixed modes
		tau_mod_Dpi1 = tau % Dpi1				# stretched periods modulo Dpi1 for gravity-dominated mixed modes
		lignes_gravity_modes = np.size(gravity_modes, axis=0)
	return freq, tau, spec_dens, tau_mod_Dpi1, lignes_gravity_modes



### Routine selecting the most significant oscillation modes: used to fit the model

def function_most_significant_modes(spec_dens, high_coef, freq, tau, Dpi1, lignes_gravity_modes):
	high_crit = spec_dens[np.argsort(spec_dens)[::-1]][2]			# selecting the power spectral density of the 3rd highest mode in the power spectrum
	high_modes_crit = np.where(spec_dens > high_coef * high_crit)
	high_freq = freq[high_modes_crit]					# frequencies for the most significant gravity-dominated mixed modes
	high_tau = tau[high_modes_crit]						# stretched periods for the most significant gravity-dominated mixed modes
	high_spec_dens = spec_dens[high_modes_crit]				# spectral densities for the most significant gravity-dominated mixed modes
	high_tau_mod_Dpi1 = high_tau % Dpi1					# stretched periods modulo Dpi1 for the most significant gravity-dominated mixed modes
	size_high_points = np.size(high_freq, axis=0)
	if size_high_points == lignes_gravity_modes:				# case where the number of selected significant modes is the same as the total number of gravity-dominated mixed modes
		high_modes_crit = np.where(spec_dens > np.median(spec_dens))[0]
		high_freq = freq[high_modes_crit]
		high_tau = tau[high_modes_crit]
		high_spec_dens = spec_dens[high_modes_crit]
		high_tau_mod_Dpi1 = high_tau % Dpi1				
		size_high_points = np.size(high_freq, axis=0)
	return high_freq, high_tau, high_spec_dens, high_tau_mod_Dpi1, size_high_points



### Routine plotting the échelle diagram of gravity-dominated mixed modes with most significant modes identified: visual tool to identify the number of visible rotational components and provide it as an input parameter to guide the fit of the model

def plot_echelle_identify_number_components(tau_mod_Dpi1, freq, spec_dens, high_tau_mod_Dpi1, high_freq, high_spec_dens, KIC, Dpi1):
	plt.figure(figsize=(8, 6))
	plt.scatter(tau_mod_Dpi1, freq, s=spec_dens)
	plt.scatter(high_tau_mod_Dpi1, high_freq, s=high_spec_dens, c='r')
	plt.xlim(min(tau_mod_Dpi1)-20, max(tau_mod_Dpi1)+20)
	plt.ylim(min(freq)-10, max(freq)+10)
	good_name = 'KIC ' + str(int(KIC)) + ' with ' + r'$\Delta \Pi_{1}$ =' + '%.2f' % Dpi1 + ' s' 
	plt.title(good_name, fontsize = 'x-large')
	plt.xlabel(r'$\tau$ mod $\Delta \Pi_{1}$', fontsize = 'x-large')
	plt.ylabel(r'$\nu \, (\mu$' + 'Hz)', fontsize = 'x-large')
	plt.show()
	return



### Routine defining the range of rotation values to be tested to fit the data

def function_range_dnurot_to_test():
	dnurot_list = []
	dnurot_min = 100		# in nHz
	dnurot_max = 1000		# in nHz
	dnurot_list.append(dnurot_min)
	dnurot_step = 20		# in nHz
	dnurot_numbers = (dnurot_max-dnurot_min)/dnurot_step
	for j in range(int(dnurot_numbers)):
		dnurot_mat = np.array(dnurot_list)
		size_dnurot_mat = np.size(dnurot_mat, axis=0)
		dnurot_list.append(dnurot_mat[size_dnurot_mat-1] + dnurot_step)
	dnurot_range = np.array(dnurot_list)
	return dnurot_range, dnurot_numbers



### Routine defining the range of period spacings values to be tested to fit the data

def function_range_Dpi1_to_test():
	Dpi1_min = Dpi1 * (1 - 0.03)	# in secs
	Dpi1_max = Dpi1 * (1 + 0.03)	# in secs
	Dpi1_step = 0.1			# in secs
	Dpi1_numbers = int((Dpi1_max - Dpi1) / Dpi1_step)
	a = []
	a.append(Dpi1)
	for j in range(1, Dpi1_numbers):
		a.append(Dpi1 - j * Dpi1_step)
		a.append(Dpi1 + j * Dpi1_step)
	Dpi1_tilt = np.array(a)
	Dpi1_tilt = Dpi1_tilt[np.argsort(Dpi1_tilt)]
	size_Dpi1_tilt = np.size(Dpi1_tilt, axis=0)
	return Dpi1_tilt, size_Dpi1_tilt



### Routine constructing synthetic frequencies and stretched periods based on given rotation value and a given period spacing Dpi1

def function_synthetic_nu_tau(nm_star, dnurot_star, Dpi1_tilt):

	## Defining periods regularly spaced and their associated frequencies

	P0_star = nm_star * Dpi1_tilt[ind]
	nu0_star = 10**6 / P0_star			# in muHz

	## Including the effect of rotation on the frequencies: perturbation generating two additional frequencies all along the spectrum

	nu_plus_star = nu0_star - dnurot_star * 10**6		# in muHz
	nu_moins_star = nu0_star + dnurot_star * 10**6		# in muHz


	## Obtaining the stretched-period spacing disturbed by rotation from the frequencies

	dtau_plus_star = Dpi1_tilt[ind] / (1 - dnurot_star / (10**(-6) * nu0_star))**2
	dtau_moins_star = Dpi1_tilt[ind] / (1 + dnurot_star / (10**(-6) * nu0_star))**2


	## Obtaining the periods disturbed by rotation from the frequencies

	P_plus_star = 10**6 / nu_plus_star
	P_moins_star = 10**6 / nu_moins_star


	## Building the stretched periods disturbed by rotation and magnetic fields, respectively, from the stretched-period spacing

	for k in range(1,n_star):
		P_plus_star[k] = P_plus_star[k-1] + dtau_plus_star[k]
		P_moins_star[k] = P_moins_star[k-1] + dtau_moins_star[k]


	## Obtaining the final frequencies disturbed by rotation from the stretched periods

	nu_plus_star = 10**6 / P_plus_star
	nu_moins_star = 10**6/ P_moins_star


	## Folding the stretched periods disturbed by rotation over a constant period spacing

	tau_mod_plus_star = P_plus_star % Dpi1_tilt[ind]
	tau_mod_moins_star = P_moins_star % Dpi1_tilt[ind]
	tau_mod_0_star = P0_star % Dpi1_tilt[ind]
	size_tau_mod_0_star = np.size(tau_mod_0_star, axis=0)
	tau_mod_plus_star = tau_mod_plus_star[np.argsort(nu_plus_star)]		# sorting the frequencies and assoiated stretched periods
	nu_plus_star = nu_plus_star[np.argsort(nu_plus_star)]
	tau_mod_moins_star = tau_mod_moins_star[np.argsort(nu_moins_star)]		# sorting the frequencies and assoiated stretched periods
	nu_moins_star = nu_moins_star[np.argsort(nu_moins_star)]
	nu_0_star = nu0_star[np.argsort(nu0_star)]					# sorting the frequencies and assoiated stretched periods
	
	## Definition of the different values of tau mod Dpi1 for the ridge unperturbed by rotation (azimuthal order m=0): testing three different values because the folded stretched-period is defined modulo Dpi1

	tau_mod_0_star_0 = np.zeros((size_tau_mod_0_star))
	tau_mod_0_star_1 = np.zeros((size_tau_mod_0_star))
	tau_mod_0_star_2 = np.zeros((size_tau_mod_0_star))
	tau_mod_0_star_0[:] = min(tau_mod_0_star)
	tau_mod_0_star_1[:] = min(tau_mod_0_star) + Dpi1_tilt[ind]	
	tau_mod_0_star_2[:] = min(tau_mod_0_star) + Dpi1_tilt[ind] / 2	
	return nu_0_star, tau_mod_0_star_0, tau_mod_0_star_1, tau_mod_0_star_2, nu_plus_star, tau_mod_plus_star, nu_moins_star, tau_mod_moins_star



### Routine selecting ranges of synthetic frequencies and folded stretched-periods close to the observed ones for plotting aspects

def function_selecting_appropriate_synthetic_nu_tau(nu_plus_star, dnu_range_boundary, nu_moins_star, nu_0_star, tau_mod_plus_star, tau_mod_moins_star, tau_mod_0_star_0, tau_mod_0_star_1, tau_mod_0_star_2):
	star_range_crit_0 = np.where(nu_plus_star >= min(freq)-dnu_range_boundary)
	nu_plus_star_range = nu_plus_star[star_range_crit_0]
	nu_moins_star_range = nu_moins_star[star_range_crit_0]
	nu_0_star_range = nu_0_star[star_range_crit_0]
	tau_mod_plus_star_range = tau_mod_plus_star[star_range_crit_0]
	tau_mod_moins_star_range = tau_mod_moins_star[star_range_crit_0]	
	tau_mod_0_star_range_0 = tau_mod_0_star_0[star_range_crit_0]	
	tau_mod_0_star_range_1 = tau_mod_0_star_1[star_range_crit_0]
	tau_mod_0_star_range_2 = tau_mod_0_star_2[star_range_crit_0]
	star_range_crit = np.where(nu_plus_star_range <= max(freq)+dnu_range_boundary)
	nu_plus_star = nu_plus_star_range[star_range_crit]
	nu_moins_star = nu_moins_star_range[star_range_crit]
	nu_0_star = nu_0_star_range[star_range_crit]
	tau_mod_plus_star = tau_mod_plus_star_range[star_range_crit]
	tau_mod_moins_star = tau_mod_moins_star_range[star_range_crit]
	tau_mod_0_star_0 = tau_mod_0_star_range_0[star_range_crit]	
	tau_mod_0_star_1 = tau_mod_0_star_range_1[star_range_crit]
	tau_mod_0_star_2 = tau_mod_0_star_range_2[star_range_crit]
	return nu_plus_star, nu_moins_star, nu_0_star, tau_mod_plus_star, tau_mod_moins_star, tau_mod_0_star_0, tau_mod_0_star_1, tau_mod_0_star_2



### Routine centering the synthetic pattern on the observed one using the most significant modes

def function_centering_synthetic_pattern(freq, high_freq, high_tau_mod_Dpi1, tau_mod_plus_interp, tau_mod_moins_interp, tau_mod_0_interp_0, tau_mod_0_interp_1, tau_mod_0_interp_2):

	## Defining the offset in folded stretched-period that needs to be applied to the synthetic pattern to test to center it on the selected position of the observed pattern: case of a pattern with two or three rotational components

	offset_ini_plus = abs(freq - high_freq[k])
	ref_point_crit_plus = np.where(offset_ini_plus == 0.)
	ref_point_plus = tau_mod_plus_interp[ref_point_crit_plus]
	offset_plus = ref_point_plus - high_tau_mod_Dpi1[k]


	## Defining the offset in folded stretched-period that needs to be applied to the synthetic pattern to test to center it on the selected position of the observed patter: case of a pattern with one rotational component

	ref_point_0 = tau_mod_0_interp_0[ref_point_crit_plus]
	offset_0 = ref_point_0 - high_tau_mod_Dpi1[k]


	## Centering the synthetic pattern on the selected position of the observed pattern: case of a pattern with two or three rotational components

	tau_mod_plus_interp_0 = tau_mod_plus_interp - offset_plus
	tau_mod_moins_interp_0 = tau_mod_moins_interp - offset_plus
	tau_mod_moins_interp_0_bis = tau_mod_moins_interp_0 - Dpi1_tilt[ind]
	tau_mod_0_interp_ini = tau_mod_0_interp_0 - offset_plus
	tau_mod_0_interp_bis = tau_mod_0_interp_1 - offset_plus
	tau_mod_0_interp_ter = tau_mod_0_interp_2 - offset_plus
	
	
	## Centering the synthetic pattern on the selected position of the observed pattern: case of a pattern with one rotational component
	
	tau_mod_0_interp = tau_mod_0_interp_0 - offset_0
	return ref_point_crit_plus, tau_mod_plus_interp_0, tau_mod_moins_interp_0, tau_mod_moins_interp_0_bis, tau_mod_0_interp_ini, tau_mod_0_interp_bis, tau_mod_0_interp_ter, tau_mod_0_interp



### Routine selecting the rotational components maximizing the number of modes aligned with the synthetic pattern

def function_selecting_optimal_rotational_components(lignes_gravity_modes, tau_mod_Dpi1, high_tau_mod_Dpi1, tau_mod_plus_interp_0, tau_mod_moins_interp_0, tau_mod_moins_interp_0_bis, tau_mod_0_interp_ini, tau_mod_0_interp_bis, tau_mod_0_interp_ter, tau_mod_0_interp, abs_dist_crit):
	count_plus_0 = 0.			# initializing the count for the number of modes aligned with the synthetic pattern on each rotational component
	count_moins_0_ini = 0.
	count_moins_0_bis = 0.
	count_0_ini = 0.
	count_0_bis = 0.
	count_0_ter = 0.
	count_0_alone = 0.
	count_high_plus = 0.			# initializing the count for the number of most significant modes aligned with the synthetic pattern on each rotational component
	count_high_moins_ini = 0.
	count_high_moins_bis = 0.
	count_high_0_ini = 0.
	count_high_0_bis = 0.
	count_high_0_ter = 0.
	count_high_0_alone = 0.
	list_chi_square_plus = []		# initializing the array containing the chi-squares for each fit computed using all the modes aligned with the synthetic pattern
	list_chi_square_moins_ini = []
	list_chi_square_moins_bis = []
	list_chi_square_0_ini = []
	list_chi_square_0_bis = []
	list_chi_square_0_ter = []
	list_chi_square_0_alone = []
	list_chi_square_plus_high = []		# initializing the array containing the chi-squares for each fit computed using only the most significant modes aligned with the synthetic pattern
	list_chi_square_moins_high_ini = []
	list_chi_square_moins_high_bis = []
	list_chi_square_0_high_ini = []
	list_chi_square_0_high_bis = []
	list_chi_square_0_high_ter = []
	list_chi_square_0_alone_high = []


	## Computing the distance between the selected mode and each rotational component

	for m in range(lignes_gravity_modes):							# loop on the number of gravity modes in the power spectrum
		abs_plus_diff = abs(tau_mod_plus_interp_0[m] - tau_mod_Dpi1[m])			# computing the distance between the selected mode and the rotational component with azimuthal order m=+1
		abs_moins_diff_0 = abs(tau_mod_moins_interp_0[m] - tau_mod_Dpi1[m])		# computing the distance between the selected mode and the rotational component with azimuthal order m=-1
		abs_moins_diff_bis = abs(tau_mod_moins_interp_0_bis[m] - tau_mod_Dpi1[m])	# computing the distance of the mode from the rotational component with azimuthal order m=-1 shifted by -Dpi1
		abs_0_diff = abs(tau_mod_0_interp_ini[m] - tau_mod_Dpi1[m])			# computing the distance between the selected mode and the rotational component with azimuthal order m=0
		abs_0_diff_bis = abs(tau_mod_0_interp_bis[m] - tau_mod_Dpi1[m])			# computing the distance between the selected mode and the rotational component with azimuthal order m=-1 shifted by +Dpi1
		abs_0_diff_ter = abs(tau_mod_0_interp_ter[m] - tau_mod_Dpi1[m])			# computing the distance between the selected mode and the rotational component with azimuthal order m=-1 shifted by +Dpi1/2
		dist_mat = np.zeros((1,6))							# initializing the array containing the distances between the selected mode and each rotational component
		dist_mat[0,0] = abs_plus_diff
		dist_mat[0,1] = abs_moins_diff_0
		dist_mat[0,2] = abs_moins_diff_bis
		dist_mat[0,3] = abs_0_diff
		dist_mat[0,4] = abs_0_diff_bis
		dist_mat[0,5] = abs_0_diff_ter


		## Selecting only the most significant modes: computing the chi-square of the fit and counting the number of modes aligned with the synthetic pattern exhibiting two or three rotational components
		
		high_aligned_crit = np.where(high_tau_mod_Dpi1 == tau_mod_Dpi1[m])[0]					# focusing only on the most significant modes
		if list(high_aligned_crit):							
			dist_high_aligned_crit = np.where(dist_mat[0,:] <= abs_dist_crit)[0]				# selecting ony cases where the selected mode is close enough from the fit
			if list(dist_high_aligned_crit):
				min_dist_high_aligned_crit = np.where(dist_mat[0,:] == min(dist_mat[0,:]))[0]		# selecting the configuration where the distance between the selected mode and the fit is minimal
				if min_dist_high_aligned_crit == 0:							# case where the mode is closest to the rotational component with azimuthal order m=+1
					count_high_plus = count_high_plus + 1
					high_plus_dist = abs(tau_mod_plus_interp_0[m] - high_tau_mod_Dpi1[high_aligned_crit])
					list_chi_square_plus_high.append(high_plus_dist**2)
				if min_dist_high_aligned_crit == 1:							# case where the mode is closest to the rotational component with azimuthal order m=-1
					count_high_moins_ini = count_high_moins_ini + 1
					high_moins_dist_ini = abs(tau_mod_moins_interp_0[m] - high_tau_mod_Dpi1[high_aligned_crit])
					list_chi_square_moins_high_ini.append(high_moins_dist_ini**2)
				if min_dist_high_aligned_crit == 2:							# case where the mode is closest to the rotational component with azimuthal order m=-1 shifted by -Dpi1
					count_high_moins_bis = count_high_moins_bis + 1
					high_moins_dist_bis = abs(tau_mod_moins_interp_0_bis[m] - high_tau_mod_Dpi1[high_aligned_crit])
					list_chi_square_moins_high_bis.append(high_moins_dist_bis**2)
				if min_dist_high_aligned_crit == 3:							# case where the mode is closest to the rotational component with azimuthal order m=0
					count_high_0_ini = count_high_0_ini + 1
					high_0_dist_ini = abs(tau_mod_0_interp_ini[m] - high_tau_mod_Dpi1[high_aligned_crit])
					list_chi_square_0_high_ini.append(high_0_dist_ini**2)
				if min_dist_high_aligned_crit == 4:							# case where the mode is closest to the rotational component with azimuthal order m=0 shifted by +Dpi1
					count_high_0_bis = count_high_0_bis + 1
					high_0_dist_bis = abs(tau_mod_0_interp_bis[m] - high_tau_mod_Dpi1[high_aligned_crit])
					list_chi_square_0_high_bis.append(high_0_dist_bis**2)
				if min_dist_high_aligned_crit == 5:							# case where the mode is closest to the rotational component with azimuthal order m=0 shifted by +Dpi1/2
					count_high_0_ter = count_high_0_ter + 1
					high_0_dist_ter = abs(tau_mod_0_interp_ter[m] - high_tau_mod_Dpi1[high_aligned_crit])
					list_chi_square_0_high_ter.append(high_0_dist_ter**2)
			count_high_0_alone = count_high_0_alone + 1							# case where the mode is closest to the rotational component with azimuthal order m=0 in the case only one rotational component is present: m=0
			high_0_dist = abs(tau_mod_0_interp[m] - high_tau_mod_Dpi1[high_aligned_crit])
			list_chi_square_0_alone_high.append(high_0_dist**2)


		## Computing the chi-square of the fit and counting the number of modes aligned with the synthetic pattern for all the gravity modes

		dist_mat_crit = np.where(dist_mat[0,:] <= abs_dist_crit)[0]						# selecting ony cases where the selected mode is close enough from the fit
		if list(dist_mat_crit):
			min_dist_mat_crit = np.where(dist_mat[0,:] == min(dist_mat[0,:]))[0]				# selecting the configuration where the distance between the selected mode and the fit is minimal
			if min_dist_mat_crit == 0:									# case where the mode is closest to the rotational component with azimuthal order m=+1
				count_plus_0 = count_plus_0 + 1
				list_chi_square_plus.append(abs_plus_diff**2)
			if min_dist_mat_crit == 1:									# case where the mode is closest to the rotational component with azimuthal order m=-1
				count_moins_0_ini = count_moins_0_ini + 1
				list_chi_square_moins_ini.append(abs_moins_diff_0**2)
			if min_dist_mat_crit == 2:									# case where the mode is closest to the rotational component with azimuthal order m=-1 shifted by -Dpi1
				count_moins_0_bis = count_moins_0_bis + 1
				list_chi_square_moins_bis.append(abs_moins_diff_bis**2)
			if min_dist_mat_crit == 3:									# case where the mode is closest to the rotational component with azimuthal order m=0
				count_0_ini = count_0_ini + 1
				list_chi_square_0_ini.append(high_0_dist**2)
			if min_dist_mat_crit == 4:									# case where the mode is closest to the rotational component with azimuthal order m=0 shifted by +Dpi1
				count_0_bis = count_0_bis + 1
				list_chi_square_0_bis.append(high_0_dist**2)
			if min_dist_mat_crit == 5:									# case where the mode is closest to the rotational component with azimuthal order m=0 shifted by +Dpi1/2
				count_0_ter = count_0_ter + 1
				list_chi_square_0_ter.append(high_0_dist**2)
		abs_0_diff = abs(tau_mod_0_interp[m] - tau_mod_Dpi1[m])				# computing the distance between the selected mode and the rotational component with azimuthal order m=0 in the case of a pattern with only one rotational component
		if abs_0_diff <= abs_dist_crit:							# selecting ony cases where the selected mode is close enough from the fit
			count_0_alone = count_0_alone + 1
			list_chi_square_0_alone.append(abs_0_diff**2)
	if list_chi_square_plus:								# summing all individual chi-squares for modes belonging to the rotational component with azimuthal order m=+1
		mat_chi_square_plus = np.array(list_chi_square_plus)
		size_plus = np.size(mat_chi_square_plus, axis=0)
		chi_square_plus = np.sum(mat_chi_square_plus)
	if list_chi_square_plus_high:								# summing all individual chi-squares for the most significant modes belonging to the rotational component with azimuthal order m=+1
		mat_chi_square_plus_high = np.array(list_chi_square_plus_high)
		chi_square_plus_high = np.sum(mat_chi_square_plus_high)
	mat_chi_square_0_alone = np.array(list_chi_square_0_alone)
	size_0_alone = np.size(mat_chi_square_0_alone, axis=0)
	chi_square_0_alone = np.sum(mat_chi_square_0_alone)				# summing all individual chi-squares for modes belonging to the rotational component with azimuthal order m=0 in the case only one the m=0 rotational component is present
	mat_chi_square_0_alone_high = np.array(list_chi_square_0_alone_high)
	chi_square_0_alone_high = np.sum(mat_chi_square_0_alone_high)			# summing all individual chi-squares for the most significant modes belonging to the rotational component with azimuthal order m=+0 in the case only the m=0 rotational component is present


	## Selecting the correct configuration for the rotational components with azimuthal order m=-1

	which_moins = np.zeros((1,1))
	if count_moins_0_ini > count_moins_0_bis:						# case where more modes belong to the rotational component with azimuthal order m=-1
		count_moins_0 = count_moins_0_ini
		count_high_moins = count_high_moins_ini
		mat_chi_square_moins = np.array(list_chi_square_moins_ini)
		size_moins = np.size(mat_chi_square_moins, axis=0)
		chi_square_moins = np.sum(mat_chi_square_moins)					# summing all individual chi-squares for modes belonging to the rotational component with azimuthal order m=-1
		mat_chi_square_moins_high = np.array(list_chi_square_moins_high_ini)
		chi_square_moins_high = np.sum(mat_chi_square_moins_high)			# summing all individual chi-squares for the most significant modes belonging to the rotational component with azimuthal order m=-1
		tau_mod_moins_interp = np.zeros((lignes_gravity_modes))
		tau_mod_moins_interp[:] = tau_mod_moins_interp_0[:]
		which_moins[:] = 1
	if count_moins_0_ini <= count_moins_0_bis:						# case where more modes belong to the rotational component with azimuthal order m=-1 shifted by -Dpi1
		count_moins_0 = count_moins_0_bis
		count_high_moins = count_high_moins_bis
		mat_chi_square_moins = np.array(list_chi_square_moins_bis)
		size_moins = np.size(mat_chi_square_moins, axis=0)
		chi_square_moins = np.sum(mat_chi_square_moins)					# summing all individual chi-squares for modes belonging to the rotational component with azimuthal order m=-1 shifted by -Dpi1
		mat_chi_square_moins_high = np.array(list_chi_square_moins_high_bis)
		chi_square_moins_high = np.sum(mat_chi_square_moins_high)			# summing all individual chi-squares for the most significant modes belonging to the rotational component with azimuthal order m=-1 shifted by -Dpi1
		tau_mod_moins_interp = np.zeros((lignes_gravity_modes))
		tau_mod_moins_interp[:] = tau_mod_moins_interp_0 - Dpi1_tilt[ind]
		which_moins[:] = 2


	## Selecting the correct configuration for the rotational components with azimuthal order m=0

	mat_count_0 = np.zeros((1,3))
	mat_count_0[0,0] = count_0_ini
	mat_count_0[0,1] = count_0_bis
	mat_count_0[0,2] = count_0_ter
	which_0 = np.zeros((1,1))
	which_0_crit = np.where(mat_count_0[0,:] == max(mat_count_0[0,:]))[0][0]	# selecting the configuration with the largest number of aligned modes
	if which_0_crit == 0:								# case where more modes belong to the rotational component with azimuthal order m=0
		count_0 = count_0_ini
		count_high_0 = count_high_0_ini
		mat_chi_square_0 = np.array(list_chi_square_0_ini)
		size_0 = np.size(mat_chi_square_0, axis=0)
		chi_square_0 = np.sum(mat_chi_square_0)					# summing all individual chi-squares for modes belonging to the rotational component with azimuthal order m=0
		mat_chi_square_0_high = np.array(list_chi_square_0_high_ini)
		chi_square_0_high = np.sum(mat_chi_square_0_high)			# summing all individual chi-squares for the most significant modes belonging to the rotational component with azimuthal order m=0
		tau_mod_0_interp = np.zeros((lignes_gravity_modes))
		tau_mod_0_interp[:] = tau_mod_0_interp_ini[:]
		which_0[:] = 1
	if which_0_crit == 1:								# case where more modes belong to the rotational component with azimuthal order m=0 shifted by +Dpi1
		count_0 = count_0_bis
		count_high_0 = count_high_0_bis
		mat_chi_square_0 = np.array(list_chi_square_0_bis)
		size_0 = np.size(mat_chi_square_0, axis=0)
		chi_square_0 = np.sum(mat_chi_square_0)					# summing all individual chi-squares for modes belonging to the rotational component with azimuthal order m=0 shifted by +Dpi1
		mat_chi_square_0_high = np.array(list_chi_square_0_high_bis)
		chi_square_0_high = np.sum(mat_chi_square_0_high)			# summing all individual chi-squares for the most significant modes belonging to the rotational component with azimuthal order m=0 shifted by +Dpi1
		tau_mod_0_interp = np.zeros((lignes_gravity_modes))
		tau_mod_0_interp[:] = tau_mod_0_interp_bis[:]
		which_0[:] = 2
	if which_0_crit == 2:								# case where more modes belong to the rotational component with azimuthal order m=0 shifted by +Dpi1/2
		count_0 = count_0_ter
		count_high_0 = count_high_0_ter
		mat_chi_square_0 = np.array(list_chi_square_0_ter)
		size_0 = np.size(mat_chi_square_0, axis=0)
		chi_square_0 = np.sum(mat_chi_square_0)					# summing all individual chi-squares for modes belonging to the rotational component with azimuthal order m=0 shifted by +Dpi1/2
		mat_chi_square_0_high = np.array(list_chi_square_0_high_ter)
		chi_square_0_high = np.sum(mat_chi_square_0_high)			# summing all individual chi-squares for the most significant modes belonging to the rotational component with azimuthal order m=0 shifted by +Dpi1/2
		tau_mod_0_interp = np.zeros((lignes_gravity_modes))
		tau_mod_0_interp[:] = tau_mod_0_interp_ter[:]
		which_0[:] = 3
	return count_plus_0, count_moins_0, count_0, count_0_alone, count_high_plus, count_high_moins, count_high_0, count_high_0_alone, which_moins, which_0, size_plus, size_moins, size_0, size_0_alone, chi_square_plus, chi_square_moins, chi_square_0, chi_square_0_alone, chi_square_plus_high, chi_square_moins_high, chi_square_0_high, chi_square_0_alone_high



### Routine appending all the synthetic configurations that match the observed one

def function_appending_matching_synthetic_patterns(count_plus_0, count_moins_0, count_0, count_0_alone, count_high_plus, count_high_moins, count_high_0, count_high_0_alone, k, j, ind, chi_square_plus_high, chi_square_moins_high, chi_square_0_high, chi_square_0_alone_high, chi_square_plus, chi_square_moins, chi_square_0, chi_square_0_alone, which_moins, which_0, ridge_list_3, ridge_list_2, ridge_list_1):

	## Synthetic configurations with three rotational components: m={-1, 0, +1}

	if count_plus_0 >= 3 and count_moins_0 >= 3 and count_0 >= 3:				# case where a solution with a synthetic pattern exhibiting three rotational components is found: more than three modes aligned with each rotational component
		if count_high_plus != 0 and count_high_moins != 0 and count_high_0 != 0:	# ensuring that there is at least one most significant mode aligned with each rotational component
			ridges_mat_3 = np.zeros((1,8))
			ridges_mat_3[0,0] = int(k)						# appending the index corresponding to the tested position of the synthetic pattern
			ridges_mat_3[0,1] = int(j)						# appending the index corresponding to the tested rotation value
			ridges_mat_3[0,2] = (chi_square_plus_high + chi_square_moins_high  + chi_square_0_high) / (count_high_plus + count_high_moins + count_high_0)		# reduced chi-square computed using only the most significant modes	
			ridges_mat_3[0,3] = int(ind)						# appending the index corresponding to the tested period spacing value Dpi1
			ridges_mat_3[0,4] = int(count_high_plus + count_high_moins + count_high_0)						# total number of most significant modes aligned with all the rotational components
			ridges_mat_3[0,5] = (chi_square_plus + chi_square_moins + chi_square_0) / (count_plus_0 + count_moins_0 + count_0)					# reduced chi-square using all gravity-dominated mixed modes
			ridges_mat_3[0,6] = which_moins[0,0]					# appending the number identifying the correct configuration for the rotational component with azimuthal order m=-1
			ridges_mat_3[0,7] = which_0[0,0]					# appending the number identifying the correct configuration for the rotational component with azimuthal order m=0
			ridge_list_3.append(ridges_mat_3[0,:])


	## Synthetic configurations with two rotational components: m={-1, +1}


	if count_plus_0 >= 3 and count_moins_0 >= 3 and count_0 < 3:			# case where a solution with a synthetic pattern exhibiting two rotational components is found: more than three modes aligned with each rotational component
		if count_high_plus != 0 and count_high_moins != 0:			# ensuring that there is at least one most significant mode aligned with each rotational component
			ridges_mat_2 = np.zeros((1,8))
			ridges_mat_2[0,0] = int(k)					# appending the index corresponding to the tested position of the synthetic pattern
			ridges_mat_2[0,1] = int(j)					# appending the index corresponding to the tested rotation value
			ridges_mat_2[0,2] = (chi_square_plus_high + chi_square_moins_high) / (count_high_plus + count_high_moins)		# reduced chi-square computed using only the most significant modes
			ridges_mat_2[0,3] = int(ind)					# appending the index corresponding to the tested period spacing value Dpi1
			ridges_mat_2[0,4] = int(count_high_plus + count_high_moins)								# total number of most significant modes aligned with all the rotational components
			ridges_mat_2[0,5] = (chi_square_plus + chi_square_moins) / (count_plus_0 + count_moins_0)				# reduced chi-square using all gravity-dominated mixed modes
			ridges_mat_2[0,6] = which_moins[0,0]				# appending the number identifying the correct configuration for the rotational component with azimuthal order m=-1
			ridges_mat_2[0,7] = 0						# appending the number identifying the correct configuration for the rotational component with azimuthal order m=0; 0 because the m=0 component is absent here
			ridge_list_2.append(ridges_mat_2[0,:])


	## Synthetic configurations with one rotational components: m=0

	if count_0_alone >= 3:								# case where a solution with a synthetic pattern exhibiting one rotational component is found: more than three modes aligned with each rotational component
		ridges_mat_0 = np.zeros((1,8))
		ridges_mat_0[0,0] = int(k)						# appending the index corresponding to the tested position of the synthetic pattern
		ridges_mat_0[0,1] = 0							# appending the index corresponding to the tested rotation value; 0 because the m=0 rotational component is not senstive to rotation
		ridges_mat_0[0,2] = chi_square_0_alone_high / count_high_0_alone	# reduced chi-square computed using only the most significant modes
		ridges_mat_0[0,3] = int(ind)						# appending the index corresponding to the tested period spacing value Dpi1
		ridges_mat_0[0,4] = int(count_high_0_alone)				# total number of most significant modes aligned with all the rotational components
		ridges_mat_0[0,5] = chi_square_0_alone / count_0_alone			# reduced chi-square computed using only the most significant modes
		ridges_mat_0[0,6] = 0							# this colum is useless in the case where only the m=0 rotational component is present; but we keep the same format than for the other configurations
		ridges_mat_0[0,7] = 0							# this colum is useless in the case where only the m=0 rotational component is present; but we keep the same format than for the other configurations
		ridge_list_1.append(ridges_mat_0[0,:])
	return



### Routine selecting the three synthetic configurations fitting best the data for each tested Dpi1 value and rotation value: configurations with one, two, and three rotational components	

def function_selecting_best_synthetic_patterns_given_Dpi1_dnurot(ridge_list_3, ridge_list_2, ridge_list_1):
	if ridge_list_3:					# case where at least one synthetic pattern exhibiting three rotational components is found
		mat_3_pos_ini = np.array(ridge_list_3)
		mat_3_pos_crit = np.where(mat_3_pos_ini[:,4] == max(mat_3_pos_ini[:,4]))[0]		# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		mat_3_pos = mat_3_pos_ini[mat_3_pos_crit,:]
		size_mat_3_pos = np.size(mat_3_pos, axis=0)
		if size_mat_3_pos == 1:
			mat_Dpi1_3.append(mat_3_pos[0,:])
		if size_mat_3_pos > 1:									# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			mat_3_pos_crit_bis = np.where(mat_3_pos[:,5] == min(mat_3_pos[:,5]))[0]		# selecting the pattern corresponding to the minimum reduced chi-square value
			mat_3_pos_bis = mat_3_pos[mat_3_pos_crit_bis,:]
			mat_Dpi1_3.append(mat_3_pos_bis[0,:])
	if ridge_list_2:					# case where at least one synthetic pattern exhibiting two rotational components is found
		mat_2_pos_ini = np.array(ridge_list_2)
		mat_2_pos_crit = np.where(mat_2_pos_ini[:,4] == max(mat_2_pos_ini[:,4]))[0]		# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		mat_2_pos = mat_2_pos_ini[mat_2_pos_crit,:]
		size_mat_2_pos = np.size(mat_2_pos, axis=0)
		if size_mat_2_pos == 1:
			mat_Dpi1_2.append(mat_2_pos[0,:])
		if size_mat_2_pos > 1:									# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			mat_2_pos_crit_bis = np.where(mat_2_pos[:,5] == min(mat_2_pos[:,5]))[0]		# selecting the pattern corresponding to the minimum reduced chi-square value
			mat_2_pos_bis = mat_2_pos[mat_2_pos_crit_bis,:]
			mat_Dpi1_2.append(mat_2_pos_bis[0,:])
	if ridge_list_1:					# case where at least one synthetic pattern exhibiting one rotational component is found
		mat_1_pos_ini = np.array(ridge_list_1)
		mat_1_pos_crit = np.where(mat_1_pos_ini[:,4] == max(mat_1_pos_ini[:,4]))[0]		# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		mat_1_pos = mat_1_pos_ini[mat_1_pos_crit,:]
		size_mat_1_pos = np.size(mat_1_pos, axis=0)
		if size_mat_1_pos == 1:
			mat_Dpi1_1.append(mat_1_pos[0,:])
		if size_mat_1_pos > 1:									# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			mat_1_pos_crit_bis = np.where(mat_1_pos[:,5] == min(mat_1_pos[:,5]))[0]		# selecting the pattern corresponding to the minimum reduced chi-square value
			mat_1_pos_bis = mat_1_pos[mat_1_pos_crit_bis,:]
			mat_Dpi1_1.append(mat_1_pos_bis[0,:])
	return



### Routine selecting the three Dpi1 values associated to the three synthetic configurations fitting best the data for each tested rotation value: configurations with one, two, and three rotational components	

def function_selecting_best_Dpi1_synthetic_patterns(mat_Dpi1_3, mat_Dpi1_2, mat_Dpi1_1):
	if mat_Dpi1_3:					# case where at least one synthetic pattern exhibiting three rotational components is found
		Dpi1_3_ini = np.array(mat_Dpi1_3)
		Dpi1_3_crit = np.where(Dpi1_3_ini[:,4] == max(Dpi1_3_ini[:,4]))[0]		# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		Dpi1_3 = Dpi1_3_ini[Dpi1_3_crit,:]
		size_Dpi1_3 = np.size(Dpi1_3, axis=0)
		if size_Dpi1_3 == 1:
			mat_rot_3.append(Dpi1_3[0,:])
		if size_Dpi1_3 > 1:								# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			Dpi1_3_crit_bis = np.where(Dpi1_3[:,5] == min(Dpi1_3[:,5]))[0]		# selecting the pattern corresponding to the minimum reduced chi-square value
			Dpi1_3_bis = Dpi1_3[Dpi1_3_crit_bis,:]
			mat_rot_3.append(Dpi1_3_bis[0,:])
	if mat_Dpi1_2:					# case where at least one synthetic pattern exhibiting two rotational components is found
		Dpi1_2_ini = np.array(mat_Dpi1_2)
		Dpi1_2_crit = np.where(Dpi1_2_ini[:,4] == max(Dpi1_2_ini[:,4]))[0]		# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		Dpi1_2 = Dpi1_2_ini[Dpi1_2_crit,:]
		size_Dpi1_2 = np.size(Dpi1_2, axis=0)
		if size_Dpi1_2 == 1:
			mat_rot_2.append(Dpi1_2[0,:])
		if size_Dpi1_2 > 1:								# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			Dpi1_2_crit_bis = np.where(Dpi1_2[:,5] == min(Dpi1_2[:,5]))[0]		# selecting the pattern corresponding to the minimum reduced chi-square value
			Dpi1_2_bis = Dpi1_2[Dpi1_2_crit_bis,:]
			mat_rot_2.append(Dpi1_2_bis[0,:])
	if mat_Dpi1_1:					# case where at least one synthetic pattern exhibiting one rotational component is found
		Dpi1_1_ini = np.array(mat_Dpi1_1)
		Dpi1_1_crit = np.where(Dpi1_1_ini[:,4] == max(Dpi1_1_ini[:,4]))[0]		# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		Dpi1_1 = Dpi1_1_ini[Dpi1_1_crit,:]
		size_Dpi1_1 = np.size(Dpi1_1, axis=0)
		if size_Dpi1_1 == 1:
			mat_rot_1.append(Dpi1_1[0,:])
		if size_Dpi1_1 > 1:								# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			Dpi1_1_crit_bis = np.where(Dpi1_1[:,5] == min(Dpi1_1[:,5]))[0]		# selecting the pattern corresponding to the minimum reduced chi-square value
			Dpi1_1_bis = Dpi1_1[Dpi1_1_crit_bis,:]
			mat_rot_1.append(Dpi1_1_bis[0,:])
	return



### Routine selecting the dnurot values associated to the three synthetic configurations fitting best the data: configurations with one, two, and three rotational components	

def function_selecting_best_dnurot_synthetic_patterns(mat_rot_3, mat_rot_2, mat_rot_1):
	if mat_rot_3:
		rot_3_ini = np.array(mat_rot_3)						# case where at least one synthetic pattern exhibiting three rotational components is found
		rot_3_ini_crit = np.where(rot_3_ini[:,4] == max(rot_3_ini[:,4]))[0]	# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		rot_3 = rot_3_ini[rot_3_ini_crit,:]
		size_rot_3 = np.size(rot_3, axis=0)
		col_3_size = np.size(rot_3, axis=1)
		if size_rot_3 == 1:
			ridge_3 = np.zeros((size_rot_3, col_3_size))
			ridge_3[:] = rot_3[:]
		if size_rot_3 > 1:							# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			ridge_3_crit = np.where(rot_3[:,5] == min(rot_3[:,5]))[0]	# selecting the pattern corresponding to the minimum reduced chi-square value
			ridge_3 = rot_3[ridge_3_crit,:]
		ridge_3 = np.c_[ridge_3, np.zeros((1))]
		ridge_3[:,col_3_size] = 3
		list_rot.append(ridge_3[0,:])
	if mat_rot_2:									# case where at least one synthetic pattern exhibiting two rotational components is found
		rot_2_ini = np.array(mat_rot_2)
		rot_2_ini_crit = np.where(rot_2_ini[:,4] == max(rot_2_ini[:,4]))[0]	# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		rot_2 = rot_2_ini[rot_2_ini_crit,:]
		size_rot_2 = np.size(rot_2, axis=0)
		col_2_size = np.size(rot_2, axis=1)
		if size_rot_2 == 1:
			ridge_2 = np.zeros((size_rot_2, col_2_size))
			ridge_2[:] = rot_2[:]
		if size_rot_2 > 1:							# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			ridge_2_crit = np.where(rot_2[:,5] == min(rot_2[:,5]))[0]	# selecting the pattern corresponding to the minimum reduced chi-square value
			ridge_2 = rot_2[ridge_2_crit,:]
		ridge_2 = np.c_[ridge_2, np.zeros((1))]
		ridge_2[:,col_2_size] = 2
		list_rot.append(ridge_2[0,:])
	if mat_rot_1:									# case where at least one synthetic pattern exhibiting one rotational component is found
		rot_1_ini = np.array(mat_rot_1)
		rot_1_ini_crit = np.where(rot_1_ini[:,4] == max(rot_1_ini[:,4]))[0]	# selecting the pattern corresponding to the maximum of modes aligned with the synthetic pattern
		rot_1 = rot_1_ini[rot_1_ini_crit,:]
		size_rot_1 = np.size(rot_1, axis=0)
		col_1_size = np.size(rot_1, axis=1)
		if size_rot_1 == 1:
			ridge_1 = np.zeros((size_rot_1, col_1_size))
			ridge_1[:] = rot_1[:]
		if size_rot_1 > 1:							# case where several patterns have the same maximum of modes aligned with the synthetic pattern
			ridge_1_crit = np.where(rot_1[:,5] == min(rot_1[:,5]))[0]	# selecting the pattern corresponding to the minimum reduced chi-square value
			ridge_1_bis = rot_1[ridge_1_crit,:]
			ridge_1 = np.zeros((1,col_1_size))
			ridge_1[:] = ridge_1_bis[0,:]
		ridge_1 = np.c_[ridge_1, np.zeros((1))]
		ridge_1[:,col_1_size] = 1	
		list_rot.append(ridge_1[0,:])	
	return



### Routine building the synthetic configuration with several rotational components fitting best the data

def function_building_best_dnurot_synthetic_pattern_several_components(crit_3, mat_rot, Dpi1_tilt, dnurot_range, nm_star, tau, freq, high_tau, high_freq):
	if list(crit_3):
		test_Dpi1_3 = Dpi1_tilt[mat_rot[crit_3,3].astype(int)]		# period spacing value Dpi1 corresponding to the pattern with three rotational components fitting best the data
		dnurot_test_3 = dnurot_range[mat_rot[crit_3,1].astype(int)]	# rotation value corresponding to the pattern with three rotational components fitting best the data
		dnurot_3 = dnurot_test_3 * 10**(-9)				# in nHz


		## Defining periods regularly spaced and their associated frequencies

		P0_3 = nm_star * test_Dpi1_3
		nu0_3 = 10**6 / P0_3				# in muHz


		## Including the effect of rotation on the frequencies: perturbation generating two additional frequencies all along the spectrum

		nu_plus_3 = nu0_3 - dnurot_3 * 10**6		# in muHz
		nu_moins_3 = nu0_3 + dnurot_3 * 10**6		# in muHz


		## Obtaining the stretched-period spacing disturbed by rotation from the frequencies

		dtau_plus_3 = test_Dpi1_3 / (1 - dnurot_3 / (10**(-6) * nu0_3))**2
		dtau_moins_3 = test_Dpi1_3 / (1 + dnurot_3 / (10**(-6) * nu0_3))**2


		## Obtaining the periods disturbed by rotation from the frequencies

		P_plus_3 = 10**6 / nu_plus_3
		P_moins_3 = 10**6 / nu_moins_3


		## Building the stretched periods disturbed by rotation and magnetic fields, respectively, from the stretched-period spacing

		for k in range(1,n_star):
			P_plus_3[k] = P_plus_3[k-1] + dtau_plus_3[k]
			P_moins_3[k] = P_moins_3[k-1] + dtau_moins_3[k]


		## Obtaining the final frequencies disturbed by rotation from the stretched periods

		nu_plus_3 = 10**6 / P_plus_3
		nu_moins_3 = 10**6/ P_moins_3
		

		## Folding the stretched periods disturbed by rotation over a constant period spacing

		tau_mod_plus_3 = P_plus_3 % test_Dpi1_3
		tau_mod_moins_3 = P_moins_3 % test_Dpi1_3
		tau_mod_0_3 = P0_3 % test_Dpi1_3
		tau_mod_plus_3 = tau_mod_plus_3[np.argsort(nu_plus_3)]		# sorting the frequencies and assoiated stretched periods
		nu_plus_3 = nu_plus_3[np.argsort(nu_plus_3)]
		tau_mod_moins_3 = tau_mod_moins_3[np.argsort(nu_moins_3)]	# sorting the frequencies and assoiated stretched periods
		nu_moins_3 = nu_moins_3[np.argsort(nu_moins_3)]
		tau_mod_0_3 = tau_mod_0_3[np.argsort(nu0_3)]			# sorting the frequencies and assoiated stretched periods
		nu_0_3 = nu0_3[np.argsort(nu0_3)]
		
		
		## Selecting the correct configuration for the m=-1 and m=0 rotational components because the folded stretched-period is defined modulo Dpi1

		if mat_rot[crit_3,6] == 2:
			tau_mod_moins_3 = tau_mod_moins_3 - test_Dpi1_3		# m=-1 rotational component
		if mat_rot[crit_3,7] == 1:
			tau_mod_0_3[:] = min(tau_mod_0_3)			# m=0 rotational component
		if mat_rot[crit_3,7] == 2:
			tau_mod_0_3[:] = min(tau_mod_0_3) + test_Dpi1_3
		if mat_rot[crit_3,7] == 3:
			tau_mod_0_3[:] = min(tau_mod_0_3) + test_Dpi1_3/2


		## Interpolating at the observed frequencies to know the folded stretched-period: necessary to fit the synthetic model

		tau_mod_plus_interp_3 = np.interp(freq, nu_plus_3, tau_mod_plus_3)
		tau_mod_moins_interp_3 = np.interp(freq, nu_moins_3, tau_mod_moins_3)
		tau_mod_0_interp_3 = np.interp(freq, nu_0_3, tau_mod_0_3)


		## Selecting ranges of synthetic frequencies and folded stretched-periods close to the observed ones for plotting aspects

		range_3_crit_0 = np.where(nu_0_3 >= min(freq)-dnu_range_boundary)
		nu_0_3_range = nu_0_3[range_3_crit_0]
		tau_mod_0_3_range = tau_mod_0_3[range_3_crit_0]
		nu_plus_3_range = nu_plus_3[range_3_crit_0]
		nu_moins_3_range = nu_moins_3[range_3_crit_0]
		tau_mod_plus_3_range = tau_mod_plus_3[range_3_crit_0]
		tau_mod_moins_3_range = tau_mod_moins_3[range_3_crit_0]
		range_3_crit = np.where(nu_0_3_range <= max(freq)+dnu_range_boundary)
		nu_0_3 = nu_0_3_range[range_3_crit]
		tau_mod_0_3 = tau_mod_0_3_range[range_3_crit]
		nu_plus_3 = nu_plus_3_range[range_3_crit]
		nu_moins_3 = nu_moins_3_range[range_3_crit]
		tau_mod_plus_3 = tau_mod_plus_3_range[range_3_crit]
		tau_mod_moins_3 = tau_mod_moins_3_range[range_3_crit]


		### Centering the synthetic pattern on the observed one using the most significant modes

		high_tau_3_mod_Dpi1 = high_tau % test_Dpi1_3
		tau_3_mod_Dpi1 = tau % test_Dpi1_3
		high_freq_crit_3 = mat_rot[crit_3,0].astype(int)
		offset_ini_0_3 = abs(freq - high_freq[high_freq_crit_3])
		ref_point_crit_0_3 = np.where(offset_ini_0_3 == 0.)
		ref_point_plus_3 = tau_mod_plus_interp_3[ref_point_crit_0_3]
		offset_plus_3 = ref_point_plus_3 - high_tau_3_mod_Dpi1[high_freq_crit_3]
		tau_mod_plus_interp_3 = tau_mod_plus_interp_3 - offset_plus_3
		tau_mod_moins_interp_3 = tau_mod_moins_interp_3 - offset_plus_3
		tau_mod_0_interp_3 = tau_mod_0_interp_3 - offset_plus_3
		tau_mod_plus_3 = tau_mod_plus_3 - offset_plus_3
		tau_mod_moins_3 = tau_mod_moins_3 - offset_plus_3
		tau_mod_0_3 = tau_mod_0_3 - offset_plus_3
	return tau_3_mod_Dpi1, high_tau_3_mod_Dpi1, tau_mod_plus_interp_3, tau_mod_moins_interp_3, tau_mod_0_interp_3, nu_plus_3, tau_mod_plus_3, nu_moins_3, tau_mod_moins_3, nu_0_3, tau_mod_0_3, test_Dpi1_3, dnurot_test_3



### Routine building the synthetic configuration with one rotational component fitting best the data

def function_building_best_dnurot_synthetic_pattern_1_component(crit_1, mat_rot, Dpi1_tilt, nm_star, tau, freq, high_tau, high_freq):
	if list(crit_1):
		test_Dpi1_1 = Dpi1_tilt[mat_rot[crit_1,3].astype(int)]		# period spacing value Dpi1 corresponding to the pattern with one rotational component fitting best the data


		## Defining periods regularly spaced and their associated frequencies

		P0_1 = nm_star * test_Dpi1_1
		nu0_1 = 10**6 / P0_1				# in muHz


		## Folding the regularly spaced periods over a constant period spacing

		tau_mod_0_1 = P0_1 % test_Dpi1_1
		nu_0_1 = nu0_1[np.argsort(nu0_1)]


		## Interpolating at the observed frequencies to know the folded stretched-period: necessary to fit the synthetic model

		tau_mod_0_interp_1 = np.interp(freq, nu_0_1, tau_mod_0_1)


		## Selecting ranges of synthetic frequencies and folded stretched-periods close to the observed ones for plotting aspects

		range_1_crit_0 = np.where(nu_0_1 >= min(freq)-dnu_range_boundary)
		nu_0_1_range = nu_0_1[range_1_crit_0]
		tau_mod_0_1_range = tau_mod_0_1[range_1_crit_0]
		range_1_crit = np.where(nu_0_1_range <= max(freq)+dnu_range_boundary)
		nu_0_1 = nu_0_1_range[range_1_crit]
		tau_mod_0_1 = tau_mod_0_1_range[range_1_crit]


		## Centering the synthetic pattern on the observed one using the most significant modes

		high_tau_1_mod_Dpi1 = high_tau % test_Dpi1_1
		tau_1_mod_Dpi1 = tau % test_Dpi1_1
		high_freq_crit_1 = mat_rot[crit_1,0].astype(int)
		offset_ini_0_1 = abs(freq - high_freq[high_freq_crit_1])
		ref_point_crit_0_1 = np.where(offset_ini_0_1 == 0.)
		ref_point_0_1 = tau_mod_0_interp_1[ref_point_crit_0_1]
		offset_0_1 = ref_point_0_1 - high_tau_1_mod_Dpi1[high_freq_crit_1]
		tau_mod_0_interp_1 = tau_mod_0_interp_1 - offset_0_1
		tau_mod_0_1 = tau_mod_0_1 - offset_0_1
	return tau_1_mod_Dpi1, high_tau_1_mod_Dpi1, nu_0_1, tau_mod_0_1, test_Dpi1_1



### Routine plotting the échelle diagram with a synthetic pattern exhibiting three rotational components: m={-1, 0, +1}

def function_plot_echelle_3_components(freq, tau_3_mod_Dpi1, spec_dens, high_freq, high_tau_3_mod_Dpi1, high_spec_dens, nu_plus_3, tau_mod_plus_3, nu_moins_3, tau_mod_moins_3, nu_0_3, tau_mod_0_3, test_Dpi1_3):
	plt.figure(figsize=(8, 6))
	plt.scatter(tau_3_mod_Dpi1, freq, s=spec_dens)
	plt.scatter(high_tau_3_mod_Dpi1, high_freq, s=high_spec_dens, c='r', label='Most significant\ngravity-dominated\nmixed modes')
	plt.scatter(tau_mod_plus_3, nu_plus_3, s=20, c='r', marker='+', label='m=+1')
	plt.scatter(tau_mod_moins_3, nu_moins_3, s=20, c='g', marker='+', label='m=-1')
	plt.scatter(tau_mod_0_3, nu_0_3, s=20, c='c', marker='+', label='m=0')
	plt.xlim(min(tau_3_mod_Dpi1)-20, max(tau_3_mod_Dpi1)+20)
	plt.ylim(min(freq)-10, max(freq)+10)
	good_name = 'KIC ' + str(int(KIC)) + ' with ' + r'$\Delta \Pi_{1}$ =' + '%.2f' % test_Dpi1_3[0] + ' s:\nfit with 3 rotational components' 
	plt.title(good_name, fontsize = 'x-large')
	plt.xlabel(r'$\tau$ mod $\Delta \Pi_{1}$', fontsize = 'x-large')
	plt.ylabel(r'$\nu \, (\mu$' + 'Hz)', fontsize = 'x-large')
	plt.legend()
	plt.savefig('./Figures_rotation_intermediate/KIC_' + str(int(KIC)) + '_3_components.pdf', format='pdf')
	return



### Routine plotting the échelle diagram with a synthetic pattern exhibiting three rotational components: m={-1, +1}

def function_plot_echelle_2_components(freq, tau_2_mod_Dpi1, spec_dens, high_freq, high_tau_2_mod_Dpi1, high_spec_dens, nu_plus_2, tau_mod_plus_2, nu_moins_2, tau_mod_moins_2, test_Dpi1_2):
	plt.figure(figsize=(8, 6))
	plt.scatter(tau_2_mod_Dpi1, freq, s=spec_dens)
	plt.scatter(high_tau_2_mod_Dpi1, high_freq, s=high_spec_dens, c='r', label='Most significant\ngravity-dominated\nmixed modes')
	plt.scatter(tau_mod_plus_2, nu_plus_2, s=20, c='r', marker='+', label='m=+1')
	plt.scatter(tau_mod_moins_2, nu_moins_2, s=20, c='g', marker='+', label='m=-1')
	plt.xlim(min(tau_2_mod_Dpi1)-20, max(tau_2_mod_Dpi1)+20)
	plt.ylim(min(freq)-10, max(freq)+10)
	good_name = 'KIC ' + str(int(KIC)) + ' with ' + r'$\Delta \Pi_{1}$ =' + '%.2f' % test_Dpi1_2[0] + ' s:\nfit with 2 rotational components' 
	plt.title(good_name, fontsize = 'x-large')
	plt.xlabel(r'$\tau$ mod $\Delta \Pi_{1}$', fontsize = 'x-large')
	plt.ylabel(r'$\nu \, (\mu$' + 'Hz)', fontsize = 'x-large')
	plt.legend()
	plt.savefig('./Figures_rotation_intermediate/KIC_' + str(int(KIC)) + '_2_components.pdf', format='pdf')
	return



### Routine plotting the échelle diagram with a synthetic pattern exhibiting one rotational component: m=0

def function_plot_echelle_1_component(freq, tau_1_mod_Dpi1, spec_dens, high_freq, high_tau_1_mod_Dpi1, high_spec_dens, nu_0_1, tau_mod_0_1, test_Dpi1_1):
	plt.figure(figsize=(8, 6))
	plt.scatter(tau_1_mod_Dpi1, freq, s=spec_dens)
	plt.scatter(high_tau_1_mod_Dpi1, high_freq, s=high_spec_dens, c='r', label='Most significant\ngravity-dominated\nmixed modes')
	plt.scatter(tau_mod_0_1, nu_0_1, s=20, c='c', marker='+', label='m=0')
	plt.xlim(min(tau_1_mod_Dpi1)-20, max(tau_1_mod_Dpi1)+20)
	plt.ylim(min(freq)-10, max(freq)+10)
	good_name = 'KIC ' + str(int(KIC)) + ' with ' + r'$\Delta \Pi_{1}$ =' + '%.2f' % test_Dpi1_1[0] + ' s:\nfit with 1 rotational component' 
	plt.title(good_name, fontsize = 'x-large')
	plt.xlabel(r'$\tau$ mod $\Delta \Pi_{1}$', fontsize = 'x-large')
	plt.ylabel(r'$\nu \, (\mu$' + 'Hz)', fontsize = 'x-large')
	plt.legend()
	plt.savefig('./Figures_rotation_intermediate/KIC_' + str(int(KIC)) + '_1_component.pdf', format='pdf')
	return



### Routine selecting the rotation and Dpi1 values associated to the synthetic pattern fitting best the data

def function_selecting_best_dnurot_Dpi1(ridges_number, test_Dpi1_3, dnurot_test_3, test_Dpi1_2, dnurot_test_2, test_Dpi1_1):
	if ridges_number == 3:				# case where the synthetic pattern fitting best the data exhibits three rotational components
		final_Dpi1 = test_Dpi1_3
		dnurot = dnurot_test_3
		dnurot_final = dnurot * 10**(-9)
	if ridges_number == 2:				# case where the synthetic pattern fitting best the data exhibits two rotational components
		final_Dpi1 = test_Dpi1_2
		dnurot = dnurot_test_2
		dnurot_final = dnurot * 10**(-9)
	if ridges_number == 1:				# case where the synthetic pattern fitting best the data exhibits one rotational component
		final_Dpi1 = test_Dpi1_1
		dnurot = np.array([0.])
		dnurot_final = dnurot * 10**(-9)
	return dnurot_final, final_Dpi1



### Routine plotting the final échelle diagram corresponding to the measured rotation value and the optimal number of rotational components

def function_plot_echelle_final(freq, spec_dens, high_freq, high_spec_dens, tau_3_mod_Dpi1, tau_2_mod_Dpi1, tau_1_mod_Dpi1, high_tau_3_mod_Dpi1, high_tau_2_mod_Dpi1, high_tau_1_mod_Dpi1, tau_mod_plus_3, tau_mod_moins_3, tau_mod_0_3, tau_mod_plus_2, tau_mod_moins_2, tau_mod_0_1, nu_plus_3, nu_moins_3, nu_0_3, nu_plus_2, nu_moins_2, nu_0_1, KIC, final_Dpi1, dnurot_final):
	plt.figure(figsize=(8, 6))
	if ridges_number == 3:							# case where the final synthetic pattern fitting best the data exhibits three rotational components
		plt.scatter(tau_3_mod_Dpi1, freq, s=spec_dens, c='gray')
		plt.scatter(high_tau_3_mod_Dpi1, high_freq, s=high_spec_dens, c='b', label='Most significant\ngravity-dominated\nmixed modes')
		plt.scatter(tau_3_mod_Dpi1 + final_Dpi1, freq, s=spec_dens, c='gray')
		plt.scatter(high_tau_3_mod_Dpi1  + final_Dpi1, high_freq, s=high_spec_dens, c='b')
		plt.scatter(tau_mod_plus_3, nu_plus_3, s=20, c='r', marker='+', label='m=+1')
		plt.scatter(tau_mod_moins_3, nu_moins_3, s=20, c='g', marker='+', label='m=-1')
		plt.scatter(tau_mod_0_3, nu_0_3, s=20, c='c', marker='+', label='m=0')
		plt.xlim(min(tau_mod_moins_3)-5, max(tau_mod_plus_3)+5)
	if ridges_number == 2:							# case where the final synthetic pattern fitting best the data exhibits two rotational components
		plt.scatter(tau_2_mod_Dpi1, freq, s=spec_dens, c='gray')
		plt.scatter(high_tau_2_mod_Dpi1, high_freq, s=high_spec_dens, c='b', label='Most significant\ngravity-dominated\nmixed modes')
		plt.scatter(tau_2_mod_Dpi1 + final_Dpi1, freq, s=spec_dens, c='gray')
		plt.scatter(high_tau_2_mod_Dpi1  + final_Dpi1, high_freq, s=high_spec_dens, c='b')
		plt.scatter(tau_mod_plus_2, nu_plus_2, s=20, c='r', marker='+', label='m=+1')
		plt.scatter(tau_mod_moins_2, nu_moins_2, s=20, c='g', marker='+', label='m=-1')
		plt.xlim(min(tau_mod_moins_2)-5, max(tau_mod_plus_2)+5)
	if ridges_number == 1:							# case where the final synthetic pattern fitting best the data exhibits one rotational component
		plt.figure(figsize=(8, 6))
		plt.scatter(tau_1_mod_Dpi1, freq, s=spec_dens, c='gray')
		plt.scatter(high_tau_1_mod_Dpi1, high_freq, s=high_spec_dens, c='b', label='Most significant\ngravity-dominated\nmixed modes')
		plt.scatter(tau_1_mod_Dpi1 + final_Dpi1, freq, s=spec_dens, c='gray')
		plt.scatter(high_tau_1_mod_Dpi1  + final_Dpi1, high_freq, s=high_spec_dens, c='r')
		plt.scatter(tau_mod_0_1, nu_0_1, s=20, c='c', marker='+', label='m=0')
		plt.xlim(min(tau_mod_0_1)-5, max(tau_mod_0_1)+5)
	plt.ylim(min(freq)-10, max(freq)+10)
	good_name = 'KIC ' + str(int(KIC)) + ': core rotation rate ' + r'$\delta \nu_{rot,core}$ =' + '%.1f' % (dnurot_final[0]*10.**9) + ' nHz\nFit with 3 rotational components'
	plt.title(good_name, fontsize = 'x-large')
	plt.xlabel(r'$\tau$ mod $\Delta \Pi_{1}$', fontsize = 'x-large')
	plt.ylabel(r'$\nu \, (\mu$' + 'Hz)', fontsize = 'x-large')
	plt.legend()
	plt.savefig('./Echelle_diagram_KIC_' + str(int(KIC)) + '.pdf', format='pdf')
	return



### Definition of needed parameters

l = 1.						# angular degree of the oscilation modes: we work with dipole modes
d_n = 6						# interval of radial orders around n_max used to calculate the pure pressure modes frequencies
x_times_back = 8.				# threshold value for the ratio between the peaks height and the local background needed to reject the null hypothesis to more than 99.9%: modes are
						# significantly detected
d = 80.						# weighting factor for the power spectral densities for plotting considerations
centroid_crit = 10**(-1); absc_crit = 3		# criteria defining whether or not points are too close in frequency and need to be centroid
high_coef = 0.3					# coefficient to define the most significant oscillations modes: the power spectral density (PSD) has to be higher than the PSD of the 3rd highest mode
						# multiplied by this coefficient
n0_star = 20; n_star = 700			# parameters to construct the frequency and stretched-period intervals of the model used to fit the data
dnu_range_boundary = 15.



### Selecting the files containing the power spectrum

path = './Stars/stretched_spectrum_*.txt'
files = sorted(glob.glob(path))
files_number = np.size(files, axis=0)



### Selecting the files containing the physical parameters needed to exploit the spectrum

list_KIC_parameters_all = []
path_parameters = './Stars/parameters_*.txt'
files_parameters = sorted(glob.glob(path_parameters))
files_parameters_number = np.size(files_parameters, axis=0)
for i in range(files_parameters_number):						# loop on the different files
	star_parameters_i = files_parameters[i]
	KIC_parameters_i = star_parameters_i.replace('./Stars/parameters_KIC_','')
	KIC_parameters_i = KIC_parameters_i.replace('.txt','')
	list_KIC_parameters_all.append(KIC_parameters_i)
KIC_parameters = np.array(list_KIC_parameters_all)



### Defining the output file to save the results

output_path = "./Results.txt"
if os.path.exists(output_path) == False:
	fichier = open(output_path, "w")
	fichier.write("KIC Dnu (muHz) nu_max (muHz) Dpi1 (s) dnurot_core (nHz) number_of_components\n")
if os.path.exists(output_path) == True:
	fichier = open(output_path, "a")			# append the results if the file already exists to not erase previous results



### Reading the power spectrum: power spectral density versus frequency and stretched period

for i in range(files_number):						# loop on the different files
	star = files[i]
	KIC = star.replace('./Stars/stretched_spectrum_KIC_','')
	KIC = KIC.replace('.txt','')
	mat = np.loadtxt(star, skiprows=1)
	freq_0 = mat[:,0]						# retrieving the frequencies
	tau_0 = mat[:,1]						# retrieving the stretched periods
	spec_dens_0 = mat[:,2]						# retrieving the power spectral densities
	size_mat = np.size(mat,axis=0)
	lignes_number = np.size(mat, axis=0)				# defining the number of lines of the file containing the power spectrum
	col_nu = np.size(mat, axis=1)					# defining the columns of lines of the file containing the power spectrum



	### Reading the physical parameters needed to exploit the spectrum

	KIC_crit = np.where(KIC_parameters == KIC)[0][0]
	KIC = float(KIC)
	desired=[11,12,13]
	with open(files_parameters[KIC_crit], 'r') as fin:
		reader=csv.reader(fin)
		result=[[file_ind for file_ind in row] for number,row in enumerate(reader) if number in desired]
		result = np.array(result)
		result_0 = result[0,0].replace('\t0.', '')
		result_0 = result_0.split(" ")
		result_1 = result[1,0].replace('\t0.', '')
		result_1 = result_1.split(" ")
		result_2 = result[2,0].replace('\t0.', '')
		result_2 = result_2.split(" ")
		mat_0_list = []
		for j in range(np.size(result_0, axis=0)):
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
	Dnu = mat_0[0]				# large frequency separation between radial pressure oscillation modes		
	nu_max = mat_0[1]			# frequency of maximum oscillation power					
	epsilon_p = mat_0[2]			# phase shift for pressure oscillation modes
	d_0_l = mat_0[3]			# small frequency separation between dipole and radial oscillation modes			
	Dpi1 = mat_1[0]				# period spacing associated to dipole gravity oscillation modes
	q = mat_1[1]				# coupling factor between the core and the envelope
	epsilon_g = mat_1[2]			# phase shift for gravity oscillation modes
	cfon = mat_2[0]				# exponents for the local background around nu_max
	expo = mat_2[1]				
	alpha = 0.015 * Dnu**(-0.32)					
	n_max = (nu_max / Dnu) - epsilon_p	# radial order corresponding to nu_max										
	B = cfon * (freq_0/nu_max)**expo	# computing the local background around nu_max



### Detecting the oscillation modes of interest

## Keeping only gravity-dominated mixed modes: removing pressure-dominated modes (functions_selecting_peaks.py)

	nu_p = function_modes_frequencies(d_n, l, n_max, epsilon_p, d_0_l, alpha, Dnu)		# calculating the pure pressure modes frequencies
	lignes_nu_p = np.size(nu_p, axis=0)
	epsilon = 0.055 * Dnu								# threshold corresponding to the interval around each the pure pressure modes frequency used to identify for pressure-dominated mixed modes
	pressure_modes_ini, B_p_modes, gravity_modes_ini, B_g_modes = function_find_pressure_gravity_modes(lignes_number, col_nu, mat, lignes_nu_p, nu_p, epsilon, B)		# defining the pressure and gravity-dominated mixed modes and their associated local background values
	lignes_pressure_modes_ini = np.size(pressure_modes_ini, axis=0)
	lignes_gravity_modes_ini = np.size(gravity_modes_ini, axis=0)


## Selecting the peaks significantly enough above the background (functions_selecting_peaks.py)

	good_values_ini_p, good_values_ini_g = function_peaks_above_noise(lignes_pressure_modes_ini, pressure_modes_ini, B_p_modes, lignes_gravity_modes_ini, gravity_modes_ini, B_g_modes, x_times_back)


## Grouping the peaks of interest to identify the oscillation modes

	lignes_pressure_modes, lignes_gravity_modes, pressure_modes, gravity_modes = function_weighting_PSD(good_values_ini_p, good_values_ini_g, col_nu)		# weighting the power spectral densities for plotting considerations
	freq, tau, spec_dens, tau_mod_Dpi1, lignes_gravity_modes = function_centroid_peaks(gravity_modes, lignes_gravity_modes, centroid_crit, absc_crit, Dpi1)		# grouping the peaks of interest to identify the oscillation modes


## Selecting the most significant oscillation modes: used to fit the model

	high_freq, high_tau, high_spec_dens, high_tau_mod_Dpi1, size_high_points = function_most_significant_modes(spec_dens, high_coef, freq, tau, Dpi1, lignes_gravity_modes)
	


### Determining the number of visible rotational components based on a plot of the échelle diagram of gravity-dominated mixed modes with most significant modes identified: future input parameter to guide the fit of the model

	plot_echelle_identify_number_components(tau_mod_Dpi1, freq, spec_dens, high_tau_mod_Dpi1, high_freq, high_spec_dens, KIC, Dpi1)		# plotting the échelle diagram
	print(">>> input x times above background (write a number and press enter, then enter 0 to validate your choice); please enter 8 in this specific case:")
	new_x_times_back = float(input())		# manually setting the optimal level above the background above which we consider the oscillation modes for the fit:
							# optimizing the visualization of the rotational components while minimizing the number of data points to use to fit the model 
	while new_x_times_back != 0:			# case where we test different values for new_x_times_back; 0 validates the input new_x_times_back value and ends the while loop
		plt.close()
		good_values_ini_p, good_values_ini_g = function_peaks_above_noise(lignes_pressure_modes_ini, pressure_modes_ini, B_p_modes, lignes_gravity_modes_ini, gravity_modes_ini, B_g_modes, new_x_times_back)
		# selecting the peaks significantly significantly enough above the background (functions_selecting_peaks.py)

		lignes_pressure_modes, lignes_gravity_modes, pressure_modes, gravity_modes = function_weighting_PSD(good_values_ini_p, good_values_ini_g, col_nu)	# weighting the power spectral
																					# densities for plotting considerations
		freq, tau, spec_dens, tau_mod_Dpi1, lignes_gravity_modes = function_centroid_peaks(gravity_modes, lignes_gravity_modes, centroid_crit, absc_crit, Dpi1)	# grouping the peaks of interest
																					# to identify the oscillation modes
		high_freq, high_tau, high_spec_dens, high_tau_mod_Dpi1, size_high_points = function_most_significant_modes(spec_dens, high_coef, freq, tau, Dpi1, lignes_gravity_modes)	# selecting the 
																							# most significant 
																							# oscillation modes:
																							# used to fit the model
		plot_echelle_identify_number_components(tau_mod_Dpi1, freq, spec_dens, high_tau_mod_Dpi1, high_freq, high_spec_dens, KIC, Dpi1)		# plotting the échelle diagram
		print(">>> input x times above background (write a number and press enter, then enter 0 to validate your choice); please enter 8 in this specific case:")
		new_x_times_back = float(input())		# manually setting the optimal level above the background above which we consider the oscillation modes for the fit
	if new_x_times_back == 0:
		plt.close()



### Defining the parameters of the model to be tested to fit the data

	dnurot_range, dnurot_numbers = function_range_dnurot_to_test() 		# range of rotation values to be tested
	Dpi1_tilt, size_Dpi1_tilt = function_range_Dpi1_to_test()		# range of period spacing values to be tested
	abs_dist_crit = Dpi1/30				# criterium in distance for the x-axis to count the total number of aligned peaks: if distance less than abs_dist_crit, points are aligned
	list_rot = []
	mat_rot_3 = []; mat_rot_2 = []; mat_rot_1 = []				# defining three arrays for rotation corresponding to the three possible numbers of rotational components to be fitted



### Fitting the data

	for j in range(int(dnurot_numbers)):				# loop on the number of rotation values to test
		print('Testing rotation value number ' + str(j+1) + ' / ' + str(int(dnurot_numbers)))
		dnurot_star = dnurot_range[j] * 10**(-9)		# selecting a rotation value to test; in nHz
		nm_star = n0_star + np.array(range(n_star))		# construction of synthetic frequency and stretched-period intervals
		mat_Dpi1_3 = []; mat_Dpi1_2 = []; mat_Dpi1_1 = []	# defining three arrays for Dpi1 corresponding to the three possible numbers of rotational components to be fitted
		for ind in range(size_Dpi1_tilt):			# loop on the number of period spacings Dpi1 values to test
			tau_mod_Dpi1 = tau % Dpi1_tilt[ind]		# defining the folded stretched-periods corresponding to the selected Dpi1 value
			high_tau_mod_Dpi1 = high_tau % Dpi1_tilt[ind]	# defining the folded stretched-periods of the most significant modes corresponding to the selected Dpi1 value


			## Construction of synthetic frequencies and stretched periods
			
			nu_0_star, tau_mod_0_star_0, tau_mod_0_star_1, tau_mod_0_star_2, nu_plus_star, tau_mod_plus_star, nu_moins_star, tau_mod_moins_star = function_synthetic_nu_tau(nm_star, dnurot_star, Dpi1_tilt)	# computing the synthetic model
			tau_mod_plus_interp = np.interp(freq, nu_plus_star, tau_mod_plus_star)			# interpolating at the observed frequencies to know the folded stretched-period:
														# necessary to fit the synthetic model
			tau_mod_moins_interp = np.interp(freq, nu_moins_star, tau_mod_moins_star)
			tau_mod_0_interp_0 = np.interp(freq, nu_0_star, tau_mod_0_star_0)
			tau_mod_0_interp_1 = np.interp(freq, nu_0_star, tau_mod_0_star_1)
			tau_mod_0_interp_2 = np.interp(freq, nu_0_star, tau_mod_0_star_2)
			nu_plus_star, nu_moins_star, nu_0_star, tau_mod_plus_star, tau_mod_moins_star, tau_mod_0_star_0, tau_mod_0_star_1, tau_mod_0_star_2 = function_selecting_appropriate_synthetic_nu_tau(nu_plus_star, dnu_range_boundary, nu_moins_star, nu_0_star, tau_mod_plus_star, tau_mod_moins_star, tau_mod_0_star_0, tau_mod_0_star_1, tau_mod_0_star_2)
# selecting ranges of synthetic frequencies and folded stretched-periods close to the observed ones for plotting aspects


			## Centering the synthetic pattern on the observed one using the most significant modes: provides a test fit to the data

			ridge_list_3 = []			# initializing the array containing the different positions of the synthetic pattern with 3 components giving a correct fit for given dnurot and Dpi1
			ridge_list_2 = []			# initializing the array containing the different positions of the synthetic pattern with 2 components giving a correct fit for given dnurot and Dpi1
			ridge_list_1 = []			# initializing the array containing the different positions of the synthetic pattern with 1 component giving a correct fit for given dnurot and Dpi1
			for k in range(size_high_points):	# loop on the number of positions of the synthetic pattern to test: corresponds to the number of most significant modes that have been selected beforehand
				ref_point_crit_plus, tau_mod_plus_interp_0, tau_mod_moins_interp_0, tau_mod_moins_interp_0_bis, tau_mod_0_interp_ini, tau_mod_0_interp_bis, tau_mod_0_interp_ter, tau_mod_0_interp = function_centering_synthetic_pattern(freq, high_freq, high_tau_mod_Dpi1, tau_mod_plus_interp, tau_mod_moins_interp, tau_mod_0_interp_0, tau_mod_0_interp_1, tau_mod_0_interp_2)	# centering the synthetic pattern on the observed one


				## Selecting the synthetic configuration that matches best the observed one: pattern exhibiting either one, two or three rotational components

				count_plus_0, count_moins_0, count_0, count_0_alone, count_high_plus, count_high_moins, count_high_0, count_high_0_alone, which_moins, which_0, size_plus, size_moins, size_0, size_0_alone, chi_square_plus, chi_square_moins, chi_square_0, chi_square_0_alone, chi_square_plus_high, chi_square_moins_high, chi_square_0_high, chi_square_0_alone_high = function_selecting_optimal_rotational_components(lignes_gravity_modes, tau_mod_Dpi1, high_tau_mod_Dpi1, tau_mod_plus_interp_0, tau_mod_moins_interp_0, tau_mod_moins_interp_0_bis, tau_mod_0_interp_ini, tau_mod_0_interp_bis, tau_mod_0_interp_ter, tau_mod_0_interp, abs_dist_crit)		# selecting the rotational components maximizing the number of modes aligned with the synthetic pattern:
											# cases with two and three rotational components
				function_appending_matching_synthetic_patterns(count_plus_0, count_moins_0, count_0, count_0_alone, count_high_plus, count_high_moins, count_high_0, count_high_0_alone, k, j, ind, chi_square_plus_high, chi_square_moins_high, chi_square_0_high, chi_square_0_alone_high, chi_square_plus, chi_square_moins, chi_square_0, chi_square_0_alone, which_moins, which_0, ridge_list_3, ridge_list_2, ridge_list_1)		# appending all the synthetic patterns that match the observed one
			function_selecting_best_synthetic_patterns_given_Dpi1_dnurot(ridge_list_3, ridge_list_2, ridge_list_1)			# selecting the three synthetic configurations fitting best the data for each tested Dpi1 value and rotation value
		function_selecting_best_Dpi1_synthetic_patterns(mat_Dpi1_3, mat_Dpi1_2, mat_Dpi1_1)		# selecting the three Dpi1 values associated to the three synthetic configurations
														# fitting best the data for each tested rotation value
	function_selecting_best_dnurot_synthetic_patterns(mat_rot_3, mat_rot_2, mat_rot_1)			# selecting the dnurot values associated to the three synthetic configurations
						
						
	## Selecting the configuration fitting best the data: exhibiting one, two, or three rotational components
	
	if list_rot:
		mat_rot = np.array(list_rot)
		crit_3 = np.where(mat_rot[:,8] == 3)[0]			# selecting the parameters of the pattern with three rotational components fitting best the data
		crit_2 = np.where(mat_rot[:,8] == 2)[0]			# selecting the parameters of the pattern with two rotational components fitting best the data
		crit_1 = np.where(mat_rot[:,8] == 1)[0]			# selecting the parameters of the pattern with one rotational component fitting best the data
		tau_3_mod_Dpi1, high_tau_3_mod_Dpi1, tau_mod_plus_interp_3, tau_mod_moins_interp_3, tau_mod_0_interp_3, nu_plus_3, tau_mod_plus_3, nu_moins_3, tau_mod_moins_3, nu_0_3, tau_mod_0_3, test_Dpi1_3, dnurot_test_3 = function_building_best_dnurot_synthetic_pattern_several_components(crit_3, mat_rot, Dpi1_tilt, dnurot_range, nm_star, tau, freq, high_tau, high_freq)	# building the
																				# synthetic configuration with three
																				# rotational components fitting best
																				# the data
		tau_2_mod_Dpi1, high_tau_2_mod_Dpi1, tau_mod_plus_interp_2, tau_mod_moins_interp_2, tau_mod_0_interp_2, nu_plus_2, tau_mod_plus_2, nu_moins_2, tau_mod_moins_2, nu_0_2, tau_mod_0_2, test_Dpi1_2, dnurot_test_2 = function_building_best_dnurot_synthetic_pattern_several_components(crit_2, mat_rot, Dpi1_tilt, dnurot_range, nm_star, tau, freq, high_tau, high_freq)	# building the
																				# synthetic configuration with two
																				# rotational components fitting best
																				# the data
		tau_1_mod_Dpi1, high_tau_1_mod_Dpi1, nu_0_1, tau_mod_0_1, test_Dpi1_1 = function_building_best_dnurot_synthetic_pattern_1_component(crit_1, mat_rot, Dpi1_tilt, nm_star, tau, freq, high_tau, high_freq)										# building the synthetic configuration with one rotational component fitting best the data
		function_plot_echelle_3_components(freq, tau_3_mod_Dpi1, spec_dens, high_freq, high_tau_3_mod_Dpi1, high_spec_dens, nu_plus_3, tau_mod_plus_3, nu_moins_3, tau_mod_moins_3, nu_0_3, tau_mod_0_3, test_Dpi1_3)										# plotting the synthetic configuration with three rotational components fitting best the data
		function_plot_echelle_2_components(freq, tau_2_mod_Dpi1, spec_dens, high_freq, high_tau_2_mod_Dpi1, high_spec_dens, nu_plus_2, tau_mod_plus_2, nu_moins_2, tau_mod_moins_2, test_Dpi1_2)
											# plotting the synthetic configuration with two rotational components fitting best the data
		function_plot_echelle_1_component(freq, tau_1_mod_Dpi1, spec_dens, high_freq, high_tau_1_mod_Dpi1, high_spec_dens, nu_0_1, tau_mod_0_1, test_Dpi1_1)	# plotting the synthetic configuration 
																					# with one rotational component
																					# fitting best the data
		plt.close('all')
	print('>>> input number of rotational components (write a number and press enter; 0 means no satisfaction with any pattern); please enter 3 in this specific case:')
	ridges_number = float(input())		# manually setting the optimal number of rotational components based on a visual inspection of the saved figures showing the fitted échelle diagrams
	dnurot_final, final_Dpi1 = function_selecting_best_dnurot_Dpi1(ridges_number, test_Dpi1_3, dnurot_test_3, test_Dpi1_2, dnurot_test_2, test_Dpi1_1)	# selecting the rotation and Dpi1 values 
																				# associated to the synthetic pattern 
																				# fitting best the data
																				
				
																				
	### Saving the results in an output file
		
	fichier.write(str(int(KIC)) + ' ' + str('%.2f' % Dnu) + ' ' + str('%.2f' % nu_max) + ' ' + str('%.2f' % final_Dpi1[0]) + ' ' + str('%.1f' % (dnurot_final[0]*10.**9)) + ' ' + str(int(ridges_number)) + '\n')



	### Plotting the final échelle diagram corresponding to the measured rotation value and the optimal number of rotational components

	function_plot_echelle_final(freq, spec_dens, high_freq, high_spec_dens, tau_3_mod_Dpi1, tau_2_mod_Dpi1, tau_1_mod_Dpi1, high_tau_3_mod_Dpi1, high_tau_2_mod_Dpi1, high_tau_1_mod_Dpi1, tau_mod_plus_3, tau_mod_moins_3, tau_mod_0_3, tau_mod_plus_2, tau_mod_moins_2, tau_mod_0_1, nu_plus_3, nu_moins_3, nu_0_3, nu_plus_2, nu_moins_2, nu_0_1, KIC, final_Dpi1, dnurot_final)
	plt.close('all')					
fichier.close()


