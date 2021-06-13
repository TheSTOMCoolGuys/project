import STOM_higgs_tools as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy as sp
import scipy.optimize as spo
import scipy.integrate as spi
import scipy.stats as sps
import math
import scipy.special as spp
from scipy.interpolate import UnivariateSpline

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

result = spo.minimize(fit_area, guess)
A2 = result.x

fit_y_values = st.get_B_expectation(bin_edges, A2, initial_lambda)

plt.plot(bin_edges, fit_y_values, label = 'Including Signal')
# ============================================================================= FIT INCLUDING SIGNAL


# ============================================================================= USING CURVEFIT SEEMS TO WORK BEST
def parameter_fit(x, A):
    fit = A*np.exp(-x/lam)
    return fit

A3, A_cov= spo.curve_fit(parameter_fit, bin_centres2, bin_heights2, guess)

fit_y_values = st.get_B_expectation(bin_edges2, A3, initial_lambda)
plt.plot(bin_edges2, fit_y_values, label = 'Curvefit')
plt.legend()
# ============================================================================= USING CURVEFIT SEEMS TO WORK BEST

# ============================================================================= Samuel's version of chi-squared fit
def exponential(x, lamb, A):
    return A * np.exp(-x/lamb)
# Note params are a list of the form [A,lambda]
# Ni are the y values and xi the corresponding x for the function
def chi_squared(params, ni, xi): #Note: ni and xi are data arrays
    pull_i = 0
    for i in range(len(ni)):
        pull_i += (ni[i] - params[1] * np.exp(-xi[i]/params[0]))**2 / ni[i]
    return pull_i

# Create a list of x,y coordinates of the the background data points in this section
# x positions are the bin centres, and y the bin heights
bin_heights_background = []
bin_centres_background = []
for i in range(len(bin_heights)):
    if bin_centres[i] < 115 or bin_centres[i] > 130: #Choosing criterion
        bin_heights_background.append(bin_heights[i])
        bin_centres_background.append(bin_centres[i])
bin_heights_background = np.array(bin_heights_background)
bin_centres_background = np.array(bin_centres_background)

# Now the data is loaded into a minimising function to find the best chi^2 value
args = (bin_heights_background, bin_centres_background)
initial_guess = np.array([30, 10000])
results = spo.minimize(chi_squared, initial_guess, args)
chi_min = results['fun']                                        # select the chi value
lamb_opt, A_opt = results['x']                                  # selects the optimum values

# plot the background data as well as the exponential with the optimal parameters
x_array = np.linspace(104, 155, 1000)
fig, ax = plt.subplots()
ax.errorbar(bin_centres, bin_heights, xerr = bin_width/2, yerr=np.sqrt(bin_heights), fmt='.', mew=0.5, lw=0.5, ms=8, capsize=1, color='black', label='Data')
ax.plot(x_array, exponential(x_array, lamb_opt, A_opt), color='red', label='Fit')
ax.set_xlabel(r'$m_{\gamma\gamma}$ (GeV/c$^2$)')
ax.set_ylabel('Number of Entries')
ax.legend(frameon=False)
ax.tick_params(direction='in',which='both')
ax.set_title(r'Using $\chi^2$ method to find best parameters for background')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.savefig('chi.eps', format='eps)
plt.show()
# ============================================================================= Samuel's version of chi-squared fit

#Question 3, 4(a)
# ============================================================================= Generate goodness and hypothesis testing data
#This question assumes n_signal = 400. It is not to be confused with the question below, where we assumes n_signal = 0.
#Therefore, the background hypothesis is rejected here, but is accepted in question 4b.
#Edit: We run 10k iterations to ensure that our goodness of fit is not affected by random uncertainties.
iterations = 100 #Original = 10000
background_hypothesis = np.empty((iterations, 3))
for i in range(iterations):
    vals = st.generate_data()
    bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bin_width = bin_edges[1]-bin_edges[0]
    #Removing the signal portion when fitting
    bin_heights_background = []
    bin_centres_background = []
    for j in range(len(bin_heights)):
        if bin_centres[j] < 115 or bin_centres[j] > 130: #Choosing criterion
            bin_heights_background.append(bin_heights[j])
            bin_centres_background.append(bin_centres[j])
    bin_heights_background = np.array(bin_heights_background)
    bin_centres_background = np.array(bin_centres_background)

    args = (bin_heights_background, bin_centres_background)
    initial_guess = np.array([30, 10000])
    results = spo.minimize(chi_squared, initial_guess, args)
    #chi_min = results['fun']
    lamb_opt, A_opt = results['x']
    goodness = st.get_B_chi(vals, (104, 155), 30, A_opt, lamb_opt)
    #Goodness measures the ratio of chi-squared value with N_dof. It is a bad fit if goodness > 1
    chi2, p_value = sps.chisquare(bin_heights, exponential(bin_centres, lamb_opt, A_opt), ddof=1)
    #We set ddof = 1 although there are two degrees of freedom because documentation has an addition -1 to correct for bias.
    background_hypothesis[i] = (goodness, chi2, p_value)
#np.savetxt('background_hypothesis.csv', background_hypothesis, delimiter=',') #Save background hypothesis data here to save time
# ============================================================================= Generate goodness and hypothesis testing data

# ============================================================================= Find sigma distribution for background-only hypothesis when n_signal=400
#iterations = 10000 #Uncomment this
#background_hypothesis = np.loadtxt('background_hypothesis.csv', delimiter=',') #Load background hypothesis data for analysis
background_hypothesis_mean = np.mean(background_hypothesis, axis=0)
background_hypothesis_std = np.std(background_hypothesis, axis=0, ddof=1)
background_hypothesis_se = background_hypothesis_std/np.sqrt(iterations)
print('The goodness of fit (reduced chi-squared value) is %f +/- %f.' %(background_hypothesis_mean[0], background_hypothesis_se[0]))
print('The chi-squared value of the background-only hypothesis is %f +/- %f.' %(background_hypothesis_mean[1], background_hypothesis_se[1]))
#The goodness of fit (reduced chi-squared value) is 3.07 +/- 0.07. The goodness of fit is bad since reduced chi-squared > 1.
#The chi-squared value of the background-only hypothesis is 85.4 +/- 0.2. It contains the same information as the reduced chi-squared.

#Let's analyse the distribution of sigma throughout these 10k simulations by plotting a histogram.
sigma_value = np.sqrt(2)*spp.erfcinv(background_hypothesis[:,2])
sigma_value_mean = np.mean(sigma_value)
sigma_value_std = np.std(sigma_value, ddof=1)
sigma_value_se = sigma_value_std / np.sqrt(iterations) #This only measures the accuracy of the mean of sigma value, not the distribution itself.
#We measure the average sigma for 10k simulations is 5.25 +/- 0.01. This means over half of the simulations will show confirmation (sigma>5) of a new particle.

#Fit a gaussian - pretty random, but it looks like one
bin_heights, bin_edges = np.histogram(sigma_value, bins=30, range=(0,10))
bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
bin_width = bin_edges[1]-bin_edges[0]
initial_guess = (5, 1, 500)
popt, pcov = spo.curve_fit(st.signal_gaus, bin_centres, bin_heights, initial_guess)
sigma_array = np.linspace(0, 10, 1001)
fig, ax = plt.subplots()
ax.hist(sigma_value, bins=30, range=(0,10), histtype='step', color='red', label='Data')
ax.plot(sigma_array, st.signal_gaus(sigma_array, *popt), color='black', label='Fit')
ax.set_xlabel('Sigma')
ax.set_ylabel('Number of Entries')
ax.set_xlim((0,10))
ax.set_ylim((0,1200))
ax.legend(frameon=False)
ax.tick_params(direction='in',which='both', axis='y')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.savefig('sigma_distribution.eps', format='eps')
plt.show()
# ============================================================================= Find sigma distribution for background-only hypothesis when n_signal=400

#Question 4(b)
# ============================================================================= Find chi-square distribution for background only hypothesis
#This part assumes n_signal = 0. It is not to be confused with the previous part, where n_signal = 400.
#Beware of long iteration time - it shall take about ten seconds for 100 iterations, but about 10-15 minutes for 10k iterations!
chi2_array = []
iterations = 100 #Original code = 10000
for j in range(iterations):
    vals = st.generate_data(0)                                                  # Generate new data
    bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)  # Take the x,y coords
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])                            # Do the formating to get it right
    bin_width = bin_edges[1]-bin_edges[0]
    #Below is the code for selecting data if n_signal != 0
    """
    bin_heights_background = []
    bin_centres_background = []
    for i in range(len(bin_heights)):
        if bin_centres[i] < 115 or bin_centres[i] > 130: #Choosing criterion
            bin_heights_background.append(bin_heights[i])
            bin_centres_background.append(bin_centres[i])
    bin_heights_background = np.array(bin_heights_background)
    bin_centres_background = np.array(bin_centres_background)
    """
    args = (bin_heights, bin_centres)                                           # Again find the optimal parameters using Chi
    initial_guess = np.array([30, 10000])
    results = spo.minimize(chi_squared, initial_guess, args)
    lamb_opt, A_opt = results['x']
    chi2, p_value = sps.chisquare(bin_heights, exponential(bin_centres, lamb_opt, A_opt), ddof=1)
    chi2_array.append(chi2)
chi2_array.sort()
# ============================================================================= Find chi-square distribution for background only hypothesis

# ============================================================================= Plot chi-square distribution for background only hypothesis
#Plot the distribution of chi-square in 10k iterations.
#np.savetxt('chisquare_distribution_nosignal.csv', chi2_array, delimiter=',') #Save chi-square distribution data here to save time
#chi2_array = np.loadtxt('chisquare_distribution_nosignal.csv', delimiter=',') #Load chi-square distribution from saved data
chi2_x = np.linspace(0, 60, 1000)
chi2_y = sps.chi2.pdf(chi2_x, 30-2) #Second argument is ddof
fig, ax = plt.subplots()
chi2_values, chi2_bins, chi2_patches = ax.hist(chi2_array, label='Data', bins=30, histtype='step', color='red')
chi2_area = sum(np.diff(chi2_bins)*chi2_values) #Scale PDF by calculating the area under the histogram
ax.plot(chi2_x, chi2_y*chi2_area, label='PDF', color='black') #N.B. PDF changes with the number of bins
ax.set_xlabel(r'$\chi^2$')
ax.set_ylabel('Number of simulations')
ax.set_xlim((0,60))
ax.legend(frameon=False)
ax.tick_params(direction='in', which='both', axis='y')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.savefig('chisquare_distribution_nosignal.eps', format='eps')
plt.show()
# ============================================================================= Plot chi-square distribution for background only hypothesis


#Question 4(c)
# ============================================================================= Obtaining expected p-values for varying number of signals
#Warning!
#Beware of long iteration time - it shall take about three hours for 1k iterations
#Do NOT execute this code block unless your computer is connected to a power source.
#And that you have plenty of time to spare.
#Now we try to vary the number of signals to find the number of signals where the p-value = 0.05
iterations = 10     #Original code = 1000
signal_min = 150    #An initial search shows that the expected p-value starts to drop below 0.1 when signal > 150. This is set to reduce iterations
signal_max = 400    #endpoint = False
step = 5            #Must be a factor of signal_max - signal_min
p_values = []
for j in range(signal_min, signal_max+1, step):
    p_value_array = []
    for k in range(iterations):
        vals = st.generate_data(j)                                                      # Like in the previous section but with different signals
        bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)      # Again take the x,y coords
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])                                # Formatting
        bin_width = bin_edges[1]-bin_edges[0]

        bin_heights_background = []
        bin_centres_background = []
        for i in range(len(bin_heights)):                                               # Now we filter out the background data
            if bin_centres[i] < 115 or bin_centres[i] > 130: #Choosing criterion
                bin_heights_background.append(bin_heights[i])
                bin_centres_background.append(bin_centres[i])
        bin_heights_background = np.array(bin_heights_background)
        bin_centres_background = np.array(bin_centres_background)

        args = (bin_heights_background, bin_centres_background)
        initial_guess = np.array([30, 10000])
        results = spo.minimize(chi_squared, initial_guess, args)                        # Chi^2 fit using only background
        lamb_opt, A_opt = results['x']
        chi2, p_value = sps.chisquare(bin_heights, exponential(bin_centres, lamb_opt, A_opt), ddof=1) # Now find the chi^2 considering the signal as well
        p_value_array.append(p_value)                                                   # P value from the chi^2 function
    p_values.append(p_value_array)                                                      #Edit: Saved full array instead of mean data for future usage.
p_values = np.array(p_values)
#np.savetxt('pvalue_against_signal.csv', p_values, delimiter=',') #Save p-values data here to save time
# ============================================================================= Obtaining expected p-values for varying number of signals

# ============================================================================= Plot the graph of expected p-values against number of signals
#Please do not execute the code block above - unless you wanna torture yourself.
#Plot a graph of p-values against number of signals.
#p_values = np.loadtxt('pvalue_against_signal.csv', delimiter=',') #Load p-values from saved data
p_values_mean = np.mean(p_values, axis=1)
p_values_std = np.std(p_values, axis=1, ddof=1)
p_values_se = p_values_std / np.sqrt(iterations)
#signal_min = 150 #Uncomment this
#signal_max = 400 #Uncomment this
signal_range = np.array(range(signal_min, signal_max+1, step))

#Let's interpolate the data.
spl = UnivariateSpline(signal_range, p_values_mean, w=1/p_values_se)
signal_array = np.linspace(145, 405, 1001)
#And find the number of signals where the p-value = 0.05|
spl_func = lambda x: spl(x) - 0.05
critical_signal = spo.fsolve(spl_func, 250)
critical_signal = int(np.round(critical_signal))
print('The number of signals where p-value = 0.05 is ', critical_signal)
#This is the second set of simulation data I have taken.
#I forget to save the full set of data for the first simulation and missed out the error bars, so I performed another one overnight.
#The first simulation result is n_signal = 255 when p-value = 0.05.
#So both 254 and 255 signals are fine. I think 255 seems more beautiful haha.


#Plotting the data
fig, ax = plt.subplots()
ax.errorbar(signal_range, p_values_mean, yerr=p_values_se, fmt='.', mew=0.5, lw=0.5, ms=8, capsize=1, color='black', label='Data')
ax.plot(signal_array, spl(signal_array), color='red', label='Interpolation')
ax.plot(np.array([145, critical_signal]), np.array([0.05, 0.05]), '--', color='black', linewidth=0.8)
ax.plot(np.array([critical_signal, critical_signal]), np.array([0, 0.05]), '--', color='black', linewidth=0.8)
ax.set_xlabel('Number of Signals')
ax.set_ylabel('Expected p-value')
ax.set_xlim((145, 405))
ax.set_ylim(0, 0.265)
handles, labels = ax.get_legend_handles_labels()                                            #Rearranging the legend order more sensibly
handles = [handles[1], handles[0]]
labels = [labels[1], labels[0]]
ax.legend(handles, labels, frameon=False)
ax.tick_params(direction='in', which='both')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
#plt.savefig('pvalue_against_signal_v2.eps', format='eps') #Save figure here
plt.show()
#Note that we averaged out the p-value over 1000 iterations for each data point. This graph contains a lot of story!
# ============================================================================= Plot the graph of expected p-values against number of signals

# ============================================================================= Find the probability of getting a hint given expected p-value = 0.05
#Now, we find the probability of getting the hint (p-value <= 0.05) over a number of iterations when the expected p-value = 0.05 (i.e. critical_signal)
#We are literally adopting the frequentist approach and rolling a million dices.
#Note that the code is very similar to how we find the chi-squared distribution.
pvalue_array = []
iterations = 100 #Original code = 10000
for j in range(iterations):
    vals = st.generate_data(critical_signal) #found to be 255
    bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bin_width = bin_edges[1]-bin_edges[0]
    #Below is the fitting code when n_signal != 0

    bin_heights_background = []
    bin_centres_background = []
    for i in range(len(bin_heights)):
        if bin_centres[i] < 115 or bin_centres[i] > 130: #Choosing criterion
            bin_heights_background.append(bin_heights[i])
            bin_centres_background.append(bin_centres[i])
    bin_heights_background = np.array(bin_heights_background)
    bin_centres_background = np.array(bin_centres_background)

    args = (bin_heights, bin_centres)
    initial_guess = np.array([30, 10000])
    results = spo.minimize(chi_squared, initial_guess, args)
    lamb_opt, A_opt = results['x']
    chi2, p_value = sps.chisquare(bin_heights, exponential(bin_centres, lamb_opt, A_opt), ddof=1)
    pvalue_array.append(p_value)
probability_hint = len([i for i in pvalue_array if i<=0.05])/iterations
#The probability of getting a hint is found to be 69.5% for 10k iterations.
#Quite like the 1-sigma range in a Gaussian distribution.
# ============================================================================= Find the probability of getting a hint given expected p-value = 0.05
# [Samuel has left the chat]
# [Dillen has entered the chat]


# Question 5(a)
# find the Chi^2  value for the background and signal with specific values
# Let's first import the data points

# ============================================================================= Function definition for background + signal model
#Question 5a: Signal Estimation (chi2 value)

#Below shows a 2D optimisation algorithm for params = [lambda, A]
def chi_squared_2D_signal(params, ni, xi, signal_amp, mu, sig):
    pull_i = 0
    for i in range(len(ni)):
        pull_i += (ni[i] - (params[1]*np.exp(-xi[i]/params[0]) + signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((xi[i] - mu)/sig, 2.)/2)))**2 / ni[i]
    return pull_i

#And here is the 5D optimisation algorithm for params = [signal_amp, mu, sig]
def chi_squared_5D_signal(params, ni, xi, lamb, A):
    pull_i = 0
    for i in range(len(ni)):
        pull_i += (ni[i] - (A*np.exp(-xi[i]/lamb) + params[0]/(np.sqrt(2.*np.pi)*params[2])*np.exp(-np.power((xi[i] - params[1])/params[2], 2.)/2)))**2 / ni[i]
    return pull_i

#Finally, the function = exponential + signal
def complete_func(x, lamb, A, signal_amp, mu, sig):
    return A * np.exp(-x/lamb) + signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
# ============================================================================= Function definition for background + signal model

# ============================================================================= Find the chi-square distribution for background + signal model (2D)
#Warning!
#Beware of long iteration time - it shall take about ten minutes for 10k iterations.
#This code block finds the chi-square distribution for a background + signal model.
#We allow 2 degrees of freedom for lambda and A.
#The number of signals here is n_signal = 400.
chi2_array = []
pvalue_array = []
iterations = 100 #Original = 10000
for j in range(iterations):
    vals = st.generate_data()
    bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bin_width = bin_edges[1]-bin_edges[0]
    args = (bin_heights, bin_centres, 700, 125, 1.5)
    initial_guess = np.array([30, 10000])
    results = spo.minimize(chi_squared_2D_signal, initial_guess, args)
    chi_min = results['fun']
    lamb_opt, A_opt = results['x']
    chi2, p_value = sps.chisquare(bin_heights, complete_func(bin_centres, lamb_opt, A_opt, 700, 125, 1.5), ddof=1) #here ddof=2, but the function +1 automatically to remove bias, so we reduce by 1.
    chi2_array.append(chi2)
    pvalue_array.append(p_value)
chi2_array.sort()
pvalue_average = np.mean(pvalue_array)
#The mean pvalue is about 0.5. This means that we can reject the background + signal hypothesis at 50% significance level.
#This significance level is too high. This means that we cannot reject the hypothesis, and it is very likely that the hypothesis is true.
#Although there is no method to prove that it is 100% true. #Answer to question 5(b) here - elaborate

#Please do not execute the code block above more than once.
#Plot the distribution of chi-square in 10k iterations.
#np.savetxt('chisquare_distribution_incsignal_2D.csv', chi2_array, delimiter=',') #Save chi-square distribution data here to save time
#chi2_array = np.loadtxt('chisquare_distribution_incsignal_2D.csv', delimiter=',') #Load chi-square distribution from saved data
chi2_x = np.linspace(0, 60, 1000)
chi2_y = sps.chi2.pdf(chi2_x, 30-2) #Second argument is ddof
fig, ax = plt.subplots()
chi2_values, chi2_bins, chi2_patches = ax.hist(chi2_array, label='Data', bins=30, histtype='step', color='red')
chi2_area = sum(np.diff(chi2_bins)*chi2_values) #Scale PDF by calculating the area under the histogram
ax.plot(chi2_x, chi2_y*chi2_area, label='PDF', color='black') #N.B. PDF changes with the number of bins
ax.set_xlabel(r'$\chi^2$')
ax.set_ylabel('Number of simulations')
ax.set_xlim((0,60))
ax.legend(frameon=False)
ax.tick_params(direction='in', which='both', axis='y')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.savefig('chisquare_distribution_incsignal_2D.eps', format='eps')
plt.show()
# ============================================================================= Find the chi-square distribution for background + signal model (2D)

#Question 5(c)
# ============================================================================= Find the chi-square distribution for background + signal model (5D)
#Warning!
#Beware of long iteration time - it shall take about ten minutes for 10k iterations.
#This code block finds the chi-square distribution for a background + signal model.
#We allow 2 degrees of freedom for lambda and A.
#The number of signals here is n_signal = 400.
#We find that allowing 5 degrees of freedom at the same time will yield very weird results.
#Let's allow 2 degrees of freedom against the background data, then 3 degrees of freedom for the signal data.
chi2_array = []
pvalue_array = []
mass_save = []
iterations = 100 #Original = 10000
for j in range(iterations):
    #Generating and pre-processing data
    vals = st.generate_data()
    bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bin_width = bin_edges[1]-bin_edges[0]
    
    #Extracting data not without signal included
    bin_heights_background = []
    bin_centres_background = []
    for i in range(len(bin_heights)):
        if bin_centres[i] < 115 or bin_centres[i] > 130: #Choosing criterion
            bin_heights_background.append(bin_heights[i])
            bin_centres_background.append(bin_centres[i])
    bin_heights_background = np.array(bin_heights_background)
    bin_centres_background = np.array(bin_centres_background)
    
    #Fitting lambda and A first - the first two degrees of freedom
    args = (bin_heights_background, bin_centres_background)
    initial_guess = np.array([30, 10000])
    results = spo.minimize(chi_squared, initial_guess, args)
    lamb_opt, A_opt = results['x']
    
    #Now fit signal_amp, mu, sig - the other three degrees of freedom
    args = (bin_heights, bin_centres, lamb_opt, A_opt)
    initial_guess = np.array([700, 125, 1.5])
    results = spo.minimize(chi_squared_5D_signal, initial_guess, args)
    chi_min = results['fun']
    popt = results['x']
    chi2, p_value = sps.chisquare(bin_heights, complete_func(bin_centres, lamb_opt, A_opt, *popt), ddof=4) #here ddof=5, but the function +1 automatically to remove bias, so we reduce by 1.
    chi2_array.append(chi2)
    pvalue_array.append(p_value)
    mass_save.append([popt[1], popt[2]])
mass_save = np.array(mass_save)
chi2_array.sort()
#Do not sort mass array! We need its correspondence to uncertainty array to do our analysis.
#np.savetxt('mass_weighted_array_incsignal_5D.csv', mass_save, delimiter=',') #Save mass data here to save time
#mass_save = np.loadtxt('mass_array_incsignal_5D.csv', delimiter=',') #Load mass from data files
mass_array = mass_save[:,0]
uncertainty_array = mass_save[:,1]
pvalue_average = np.mean(pvalue_array)
#The mean pvalue is about 0.5 (0.49). This means that we can reject the background + signal hypothesis at 50% significance level.
#This significance level is too high. This means that we cannot reject the hypothesis, and it is very likely that the hypothesis is true.
#Although there is no method to prove that it is 100% true.

#Calculate weighted average and its uncertainty according to StoM PS3 Q1(a). Refer to that problem sheet!
mass_weighted_average = np.sum(mass_array/uncertainty_array**2)/np.sum(1/uncertainty_array**2)
#We find mass_average = 124.98 GeV. Correct!
mass_weighted_average_uncertainty = 1/np.sqrt(np.sum(1/uncertainty_array**2))
#Uncertainty is found to be 0.02 GeV. It is small because we have done lots of iterations.
#After 10k simulations, we conclude the Higgs mass to be 124.98 +/- 0.02 GeV/c^2
# ============================================================================= Find the chi-square distribution for background + signal model (5D)

# ============================================================================= Plot the chi-square distribution for background + signal model (5D)
#Please do not execute the code block above more than once.
#Plot the distribution of chi-square in 10k iterations.
#np.savetxt('chisquare_distribution_incsignal_5D.csv', chi2_array, delimiter=',') #Save chi-square distribution data here to save time
#chi2_array = np.loadtxt('chisquare_distribution_incsignal_5D.csv', delimiter=',') #Load chi-square distribution from saved data
chi2_x = np.linspace(0, 60, 1000)
chi2_y = sps.chi2.pdf(chi2_x, 30-5) #Second argument is ddof
fig, ax = plt.subplots()
chi2_values, chi2_bins, chi2_patches = ax.hist(chi2_array, label='Data', bins=30, histtype='step', color='red')
chi2_area = sum(np.diff(chi2_bins)*chi2_values) #Scale PDF by calculating the area under the histogram
ax.plot(chi2_x, chi2_y*chi2_area, label='PDF', color='black') #N.B. PDF changes with the number of bins
ax.set_xlabel(r'$\chi^2$')
ax.set_ylabel('Number of simulations')
ax.set_xlim((0,60))
ax.legend(frameon=False)
ax.tick_params(direction='in', which='both', axis='y')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.savefig('chisquare_distribution_incsignal_5D.eps', format='eps')
plt.show()
# ============================================================================= Plot the chi-square distribution for background + signal model (5D)

#Conclusion and wrapping up
# ============================================================================= Plot the famous Higgs boson graph to conclude our mighty analysis
#Final results: Play with it as you want! It executes quickly luckily.
vals = st.generate_data()
bin_heights, bin_edges = np.histogram(vals, range = [104, 155], bins = 30)
bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
bin_width = bin_edges[1]-bin_edges[0]
#Choosing data again
bin_heights_background = []
bin_centres_background = []
for i in range(len(bin_heights)):
    if bin_centres[i] < 115 or bin_centres[i] > 130: #Choosing criterion
        bin_heights_background.append(bin_heights[i])
        bin_centres_background.append(bin_centres[i])
bin_heights_background = np.array(bin_heights_background)
bin_centres_background = np.array(bin_centres_background)

#Fitting the background (now extracting errors as well)
#The error is used to plot boundaries in the background signal
initial_guess = np.array([30, 10000])
popt, pcov = spo.curve_fit(exponential, bin_centres_background, bin_heights_background, initial_guess)
perr = np.sqrt(np.diag(pcov))
lamb_opt, A_opt = popt[0], popt[1]
lamb_err, A_err = perr[0], perr[1]
y_background_err = 1/np.sqrt(np.sum(1/bin_heights)) #Refer to StoM PS3 Q1(a) again for this formula.
#Fitting the signal here
args = (bin_heights, bin_centres, lamb_opt, A_opt)
initial_guess = np.array([700, 125, 1.5])
results = spo.minimize(chi_squared_5D_signal, initial_guess, args)
signal_amp_opt, mu_opt, sig_opt = results['x']

#Plot chi-squared minimization result
mu_array = np.linspace(104, 155, 1001)
ddof_y = np.full(len(mu_array), 30-5)
fig, ax = plt.subplots()
ax.plot(mu_array, chi_squared_5D_signal((signal_amp_opt, mu_array, sig_opt), bin_heights, bin_centres, lamb_opt, A_opt), color='black', label='Data')
ax.plot(mu_array, ddof_y, color='red', label='DDOF')
ax.set_xlabel(r'$m_{\gamma\gamma}$ (GeV/c$^2$)')
ax.set_ylabel(r'$\chi^2$')
ax.set_xlim((104, 155))
ax.set_ylim((0, None))
ax.legend(frameon=False)
ax.tick_params(direction='in',which='both')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
#plt.savefig('chi2_against_mass_v2.eps', format='eps')
plt.show()

#Plotting the graph and having lots of paella
x_final_array = np.linspace(104, 155, 1001)
y_final_background = exponential(x_final_array, lamb_opt, A_opt)
fig, ax = plt.subplots()
ax.plot(x_final_array, y_final_background, '--', color='red', label='B')
ax.plot(x_final_array, complete_func(x_final_array, lamb_opt, A_opt, signal_amp_opt, mu_opt, sig_opt), color='red', label='B+S (fit)')
#ax.fill_between(x_final_array, y_final_background - y_background_err, y_final_background + y_background_err, color='yellow', label=r'B error (1$\sigma$)')
ax.fill_between(x_final_array, y_final_background - 2*y_background_err, y_final_background + 2*y_background_err, color='lawngreen', label=r'B error (2$\sigma)$')
ax.fill_between(x_final_array, y_final_background - y_background_err, y_final_background + y_background_err, color='yellow', label=r'B error (1$\sigma$)')
ax.errorbar(bin_centres, bin_heights, xerr = bin_width/2, yerr=np.sqrt(bin_heights), fmt='.', mew=0.5, lw=0.5, ms=8, capsize=1, color='black', label='Data')
ax.set_xlabel(r'$m_{\gamma\gamma}$ (GeV/c$^2$)')
ax.set_ylabel('Number of Entries')
ax.set_xlim((104, 155))
ax.set_ylim((0, 2000))
handles, labels = ax.get_legend_handles_labels()
handles = [handles[4], handles[1], handles[0], handles[3], handles[2]]
labels = [labels[4], labels[1], labels[0], labels[3], labels[2]]
ax.legend(handles, labels, frameon=False)
ax.tick_params(direction='in',which='both')
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
#plt.savefig('Higgs_boson_plot_v2.eps', format='eps')
plt.show()
# ============================================================================= Plot the famous Higgs boson graph to conclude our mighty analysis
