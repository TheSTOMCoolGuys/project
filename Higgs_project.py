import STOM_higgs_tools as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy as sp
import scipy.optimize
import scipy.integrate
import math

vals = st.generate_data()

fig, ax = plt.subplots()
bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)
bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
bin_width = bin_edges[1]-bin_edges[0]
ax.errorbar(bin_centres, bin_heights, xerr = bin_width/2, yerr=np.sqrt(bin_heights), fmt='.', mew=0.5, lw=0.5, ms=8, capsize=1, color='black', label='Data')
ax.set_xlabel(r'$m_{\gamma\gamma}$ (GeV/c$^2$)')
ax.set_ylabel('Number of Entries')
ax.legend(frameon=False)
ax.tick_params(direction='in',which='both')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.show()

#Question 2
#a.
vals = np.array(vals)
#Cutting off values higher than 120
cut_values = vals[(vals < 120)]
#Using the mean to find lambda
initial_lambda = np.mean(cut_values)

# ============================================================================= I DON'T KNOW IF DOING THIS IS OK
#Cutting off the signal
low_cutoff_vals = vals[(vals < 120)]
high_cutoff_vals = vals[(vals > 130)]
advanced_cut_values = np.concatenate((low_cutoff_vals, high_cutoff_vals))
#Again using mean
slightly_more_advanced_lambda = np.mean(advanced_cut_values)
# ============================================================================= I DON'T KNOW IF DOING THIS IS OK

lam = initial_lambda


# ============================================================================= TAKES ALL OF THE <120 MeV VALUES INTO ACCOUNT BUT THE FIT DOESN'T WORK GREAT AND DEPENDS ON BINS
#Making another histogram, but this one takes all the <120 MeV results
# numbins = 120/bin_width
# bin_heights2, bin_edges2 = np.histogram(vals, range = [1, 120], bins = 70)
# bin_centres2 = 0.5*(bin_edges2[1:]+bin_edges2[:-1])
# bin_width2 = bin_edges2[1]-bin_edges2[0]

# guess = (80000)


# histogram_area = sum(bin_width2*bin_heights2)

# def fit_area(A):
#     area = sp.integrate.quad(lambda x : A*(math.exp(-x/lam)), 1, 120)
#     return abs(area[0]-histogram_area)

# result = sp.optimize.minimize(fit_area, guess)
# A1 = result.x

# fit_y_values = st.get_B_expectation(bin_edges2, A1, initial_lambda)
# ax.errorbar(bin_centres2, bin_heights2, xerr = bin_width2/2, yerr=np.sqrt(bin_heights2), fmt='.', mew=0.5, lw=0.5, ms=8, capsize=1, color='blue', label='Data')
# # plt.xlim(104,155)
# # plt.ylim(0,2000)
# plt.plot(bin_edges2, fit_y_values, label = 'With <120MeV')

# ============================================================================= TAKES ALL OF THE <120 MeV VALUES INTO ACCOUNT BUT THE FIT DOESN'T WORK GREAT AND DEPENDS ON BINS






# ============================================================================= TAKES THE <120 MeV BUT ONLY IN OUR RANGE
bin_heights2, bin_edges2 = np.histogram(vals, range = [104, 121], bins = 10)
bin_centres2 = 0.5*(bin_edges2[1:]+bin_edges2[:-1])
bin_width3 = bin_edges2[1]-bin_edges2[0]

guess = (80000)

histogram_area = sum(bin_width3*bin_heights2)

def fit_area(A):
    area = sp.integrate.quad(lambda x : A*(math.exp(-x/lam)), 104, 121)
    return abs(area[0]-histogram_area)

result = sp.optimize.minimize(fit_area, guess)
A1 = result.x

fit_y_values = st.get_B_expectation(bin_edges, A1, initial_lambda)

plt.plot(bin_edges, fit_y_values, label = 'Excluding Signal, Only <MeV in Visual Range')
# ============================================================================= TAKES THE <120 MeV BUT ONLY IN OUR RANGE






#Making another histogram, but this one takes the <120 MeV results and >130 MeV results in our range

bin_heights2, bin_edges2 = np.histogram(advanced_cut_values, range = [104, 155], bins = 30)
bin_centres2 = 0.5*(bin_edges2[1:]+bin_edges2[:-1])
bin_width2 = bin_edges2[1]-bin_edges2[0]
#Removing those two values that don't work cos they include part of the signal
indexremove1 = np.where(bin_centres2 == 130.35)
bin_centres2 = np.delete(bin_centres2, indexremove1)
bin_heights2 = np.delete(bin_heights2, indexremove1)

indexremove2 = np.where(bin_centres2 == 120.15)
bin_centres2 = np.delete(bin_centres2, indexremove2)
bin_heights2 = np.delete(bin_heights2, indexremove2)

# ============================================================================= THIS IS FOR REMOVING THE SECTIONS OF THE SIGNAL, WHERE THE COUNT IS 0 AS WE CUT THEM OFF
for i in range(len(bin_heights2)):
    try:
        if bin_heights2[i] == 0:
            bin_centres2 = np.delete(bin_centres2, i)
            bin_heights2 = np.delete(bin_heights2, i)
    except IndexError:
        pass
for i in range(len(bin_heights2)):
    try:
        if bin_heights2[i] == 0:
            bin_centres2 = np.delete(bin_centres2, i)
            bin_heights2 = np.delete(bin_heights2, i)
    except IndexError:
        pass
for i in range(len(bin_heights2)):
    try:
        if bin_heights2[i] == 0:
            bin_centres2 = np.delete(bin_centres2, i)
            bin_heights2 = np.delete(bin_heights2, i)
    except IndexError:
        pass
# ============================================================================= IDK WHY I HAVE TO LOOP IT 3 TIMES FOR IT TO WORK BUT IT JUST DOES

# ============================================================================= THIS DOESN'T REALLY WORK BECAUSE WE DON'T HAVE THE BACKGROUND POINTS IN THE SIGNAL
guess = (80000)
#
# histogram_area = sum(bin_width2*bin_heights2)
#
# def fit_area(A):
#     area = (sp.integrate.quad(lambda x : A*(math.exp(-x/lam)), 104, 120.15-bin_width2/2)) + sp.integrate.quad(lambda x : A*(math.exp(-x/lam)), 130.35+bin_width2/2, 155)
#     return abs(area[0]-histogram_area)
#
# result = sp.optimize.minimize(fit_area, guess)
# A = result.x
#
# fit_y_values = st.get_B_expectation(bin_edges, A, initial_lambda)
# ax.errorbar(bin_centres2, bin_heights2, xerr = bin_width2/2, yerr=np.sqrt(bin_heights2), fmt='.', mew=0.5, lw=0.5, ms=8, capsize=1, color='black', label='Data')
#
# plt.plot(bin_edges, fit_y_values)
#
#
ax.errorbar(bin_centres2, bin_heights2, xerr = bin_width2/2, yerr=np.sqrt(bin_heights2), fmt='.', mew=0.5, lw=0.5, ms=8, capsize=1, color='red', label='Data Excluding Signal')
# ============================================================================= THIS DOESN'T REALLY WORK BECAUSE WE DON'T HAVE THE BACKGROUND POINTS IN THE SIGNAL

# ============================================================================= FIT INCLUDING SIGNAL
histogram_area = sum(bin_width*bin_heights)

def fit_area(A):
     area = sp.integrate.quad(lambda x : A*(math.exp(-x/lam)), 104, 155)
     return abs(area[0]-histogram_area)

result = sp.optimize.minimize(fit_area, guess)
A2 = result.x

fit_y_values = st.get_B_expectation(bin_edges, A2, initial_lambda)

plt.plot(bin_edges, fit_y_values, label = 'Including Signal')
# ============================================================================= FIT INCLUDING SIGNAL


# ============================================================================= USING CURVEFIT SEEMS TO WORK BEST
def parameter_fit(x, A):
    fit = A*np.exp(-x/lam)
    return fit

A3, A_cov= sp.optimize.curve_fit(parameter_fit, bin_centres2, bin_heights2, guess)

fit_y_values = st.get_B_expectation(bin_edges2, A3, initial_lambda)
plt.plot(bin_edges2, fit_y_values, label = 'Curvefit')
plt.legend()
# ============================================================================= USING CURVEFIT SEEMS TO WORK BEST

# ============================================================================= Samuel's version of chi-squared fit
def exponential(x, lamb, A):
    return A * np.exp(-x/lamb)
def chi_squared(params, ni, xi): #Note: ni and xi are data arrays
    pull_i = 0
    for i in range(len(ni)):
        pull_i += (ni[i] - params[1] * np.exp(-xi[i]/params[0]))**2 / ni[i]
    return pull_i

bin_heights_background = []
bin_centres_background = []
for i in range(len(bin_heights)):
    if bin_centres[i] < 115 or bin_centres[i] > 130: #Choosing criterion
        bin_heights_background.append(bin_heights[i])
        bin_centres_background.append(bin_centres[i])
bin_heights_background = np.array(bin_heights_background)
bin_centres_background = np.array(bin_centres_background)

args = (bin_heights_background, bin_centres_background)
initial_guess = np.array([30, 10000])
results = scipy.optimize.minimize(chi_squared, initial_guess, args)
chi_min = results['fun']
lamb_opt, A_opt = results['x']

x_array = np.linspace(104, 155, 1000)
fig, ax = plt.subplots()
ax.errorbar(bin_centres, bin_heights, xerr = bin_width/2, yerr=np.sqrt(bin_heights), fmt='.', mew=0.5, lw=0.5, ms=8, capsize=1, color='black', label='Data')
ax.plot(x_array, exponential(x_array, lamb_opt, A_opt), color='red', label='Fit')
ax.set_xlabel(r'$m_{\gamma\gamma}$ (GeV/c$^2$)')
ax.set_ylabel('Number of Entries')
ax.legend(frameon=False)
ax.tick_params(direction='in',which='both')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.savefig('chi.eps', format='eps)
plt.show()
# ============================================================================= Samuel's version of chi-squared fit

# ============================================================================= Find goodness and hypothesis testing
goodness = st.get_B_chi(vals, (104, 155), 30, A_opt, lamb_opt)
#Goodness measures the ratio of chi-squared value with N_dof. It is a bad fit since goodness > 1
chi2, p_value = sps.chisquare(bin_heights, exponential(bin_centres, lamb_opt, A_opt), ddof=1)
#We set ddof = 1 although there are two degrees of freedom because documentation has an addition -1 to correct for bias.
#The p-value is in the order of magnitude of 10^-7. There is a very small possibility to getting the observed or an even worse value.
#Therefore, we may reject this hypothesis at the 5e-7 = 5e-5% = 0.00005% significance level.
# ============================================================================= Find goodness and hypothesis testing

# ============================================================================= Performing multiple iterations to find chi-square distribution
#Warning!
#Beware of long iteration time - it shall take about ten seconds for 100 iterations, but about 10-15 minutes for 10k iterations!
chi2_array = []
iterations = 100 #It should be 10000, but
for j in range(iterations):
    vals = st.generate_data()
    bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bin_width = bin_edges[1]-bin_edges[0]
    bin_heights_background = []
    bin_centres_background = []
    for i in range(len(bin_heights)):
        if bin_centres[i] < 115 or bin_centres[i] > 130: #Choosing criterion
            bin_heights_background.append(bin_heights[i])
            bin_centres_background.append(bin_centres[i])
    bin_heights_background = np.array(bin_heights_background)
    bin_centres_background = np.array(bin_centres_background)

    args = (bin_heights_background, bin_centres_background)
    initial_guess = np.array([30, 10000])
    results = spo.minimize(chi_squared, initial_guess, args)
    lamb_opt, A_opt = results['x']
    chi2, p_value = sps.chisquare(bin_heights, exponential(bin_centres, lamb_opt, A_opt), ddof=1)
    chi2_array.append(chi2)
chi2_array.sort()
# ============================================================================= Performing multiple iterations to find chi-square distribution

# ============================================================================= Plotting the data distribution vs the expected distribution for ddof=28
#Please do not execute the code block above more than once, don't spend your life waiting for your computer to get hot.
#Plot the distribution of chi-square in 10k iterations.
#np.savetxt('chisquare_distribution.csv', chi2_array, delimiter=',') #Save chi-square distribution data here to save time
#chi2_array = np.loadtxt('chisquare_distribution.csv', delimiter=',') #Load chi-square distribution from saved data
chi2_x = np.linspace(0, 160, 1000)
chi2_y = sps.chi2.pdf(chi2_x, 30-2) #Second argument is ddof
fig, ax = plt.subplots()
chi2_values, chi2_bins, chi2_patches = ax.hist(chi2_array, label='Distribution', bins=30, histtype='step', color='red')
chi2_area = sum(np.diff(chi2_bins)*chi2_values) #Scale PDF by calculating the area under the histogram
ax.plot(chi2_x, chi2_y*chi2_area, label='PDF (ddof=28)', color='black') #N.B. PDF changes with the number of bins
ax.set_xlabel(r'$\chi^2$')
ax.set_ylabel('Number of simulations')
ax.set_xlim((0,150))
ax.legend(frameon=False)
ax.tick_params(direction='in', which='both', axis='y')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.savefig('chisquare_distribution.eps', format='eps)
plt.show()
# ============================================================================= Plotting the data distribution vs the expected distribution for ddof=28
