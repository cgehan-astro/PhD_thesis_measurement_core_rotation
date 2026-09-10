### Create the appropriate environment with anaconda: conda env create -f environment.yml
### python=3.9.25, ipython=8.15


import numpy as np
import pylab as plt
import matplotlib as mpl
import scipy.stats
from scipy.optimize import curve_fit


## Function power of ten to be fitted

def power_of_ten(N, a, b):
    return 10**(a*N+b)


## Defining some constants

Teff_sol = 5777.
nu_max_sol = 3050.
Dnu_sol = 135.5


## Definition of the color map to be used in the plots

cm = mpl.cm.jet


## Reading the physical parameters from the results file

fichier = np.loadtxt("Core_rotation_measurements.txt", skiprows=3)
KIC_file = fichier[:,0]
Dnu_file = fichier[:,1]
nu_max_file = fichier[:,2]
Dpi1_file = fichier[:,3]
dnurot_file = fichier[:,4]	# core rotation rate


## Selecting valid data points: non-zero core rotation rate

dnurot_crit = np.where(dnurot_file != 0.)[0]
KIC = KIC_file[dnurot_crit] 
Dnu = Dnu_file[dnurot_crit]
nu_max = nu_max_file[dnurot_crit]
Dpi1 = Dpi1_file[dnurot_crit]
dnurot = dnurot_file[dnurot_crit]
size_dnurot = np.size(dnurot, axis=0)


## Defining N, a proxy of evolution, based on scaling relations

N = 10**6 * Dnu / (Dpi1 * nu_max**2)


## Retrieving another physical parameter from another file: the effective temperature

Teff_file = np.loadtxt('APOKASC.txt', usecols=[0,43,47,49,55,56,97])
KIC_file = fichier[dnurot_crit,0]
size_KIC_file = np.size(KIC_file, axis=0)

list_Teff = []
for i in range(size_KIC_file):
	good_star_crit = np.where(Teff_file[:,0] == KIC_file[i])[0]
	if np.size(good_star_crit, axis=0) > 1:
		good_star_crit_ini = np.zeros((1))
		good_star_crit_ini[:] = good_star_crit[0]
		good_star_crit = good_star_crit_ini.astype(int)
	if list(good_star_crit):			# need to rank the dfferent measurements depending on their origin
		if Teff_file[good_star_crit,3] > 0:
			good_Teff = Teff_file[good_star_crit,3]
		if Teff_file[good_star_crit,2] > 0:
			good_Teff = Teff_file[good_star_crit,2]
		if Teff_file[good_star_crit,1] > 0:
			good_Teff = Teff_file[good_star_crit,1]
		if Teff_file[good_star_crit,6] > 0:
			good_Teff = Teff_file[good_star_crit,6]
		if Teff_file[good_star_crit,4] > 0:
			good_Teff = Teff_file[good_star_crit,4]
		if Teff_file[good_star_crit,5] > 0:
			good_Teff = Teff_file[good_star_crit,5]
	else:
		good_Teff = np.zeros((1))
		good_Teff[0] = 4800 * (nu_max[i]/40)**(0.06)
	list_Teff.append(good_Teff[0])
Teff = np.array(list_Teff)	


## Estimating the radius and mass based on scaling relations

R_over_R_sol = (nu_max / nu_max_sol) * (Dnu / Dnu_sol)**(-2) * (Teff / Teff_sol)**(1/2)
M_over_M_sol = (nu_max / nu_max_sol)**3 * (Dnu / Dnu_sol)**(-4) * (Teff / Teff_sol)**(3/2)


## Computing the Pearson correlation coefficient between mass and radius

pearson_M_R = scipy.stats.pearsonr(R_over_R_sol, M_over_M_sol)
pearson_M_R_coeff = pearson_M_R[0]
p_value_pearson_M_R_coeff = pearson_M_R[1]


## Plotting mass versus radius

plt.figure()
plt.scatter(R_over_R_sol, M_over_M_sol, alpha=0.5)
plt.annotate('Pearson correlation coefficient: ' + r'$r = $' + str(np.round(pearson_M_R_coeff,3)) + '\np-value: ' + r'$p = $' + str(np.round(p_value_pearson_M_R_coeff*10.**53,3)) + r'$\times 10^{-53}$', xy=(4, 2.1), xytext=(4, 2.1), color='k')#, fontsize='x-large')
plt.xlabel(r'$R / R_{\odot}$', fontsize = 'x-large')
plt.ylabel(r'$M / M_{\odot}$', fontsize = 'x-large')
plt.savefig('./M_R.pdf', format='pdf')
plt.close()


## Fitting the evolution of the core rotation with a power-of-ten function
    
fit_coeff_N, fit_pcov_N = curve_fit(power_of_ten, N, dnurot)

fit_coeff_N_factor = fit_coeff_N[0]
fit_coeff_N_constant = fit_coeff_N[1]

sigma_fit_coeff_N_factor = np.sqrt(np.diag(fit_pcov_N))[1]	# uncertainties on the fitted parameters
sigma_fit_coeff_N_constant = np.sqrt(np.diag(fit_pcov_N))[0]


## Plotting the evolution of the core rotation with mass represented by a color bar

absc_N = np.linspace(1., 28., 10)		# interval over which plotting the fit
fig, ax = plt.subplots()
sc = ax.scatter(N[np.argsort(M_over_M_sol)], dnurot[np.argsort(M_over_M_sol)], c=M_over_M_sol[np.argsort(M_over_M_sol)], vmin=1, vmax=2.5, s=70, cmap=cm, marker='^', edgecolors='k')		# plotting lower mass values first because larger values are rarer
ax.plot(absc_N, power_of_ten(absc_N, fit_coeff_N_factor, fit_coeff_N_constant), 'k--', linewidth=2)
ax.annotate(r'$\delta \nu_{rot,core} \propto \mathcal{N}^{-6.78 \times 10^{-4} \pm 0.014}$', xy=(1.0, 1200), xytext=(1.0, 1200), color='k', fontsize='12')
cax = fig.add_axes([0.5, 0.8, 0.3, 0.04])	# plotting the color bar
fig.colorbar(sc, cax=cax, ticks=[1, 1.3, 1.6, 1.9, 2.2, 2.5], orientation='horizontal', label='M/M$_{\odot}$')
ax.set_yscale('log')	# y-axis in logarithmic scale
ax.set_xlim(0,30)
ax.set_ylim(60,3200)
ax.set_xlabel(r'$\mathcal{N}$', fontsize = 'x-large')
ax.set_ylabel(r'$\delta \nu_{rot,core}$ (nHz)', fontsize = 'x-large')
plt.savefig('./Evolution_core_rotation_colored_with_mass.pdf', format='pdf')
plt.close()

