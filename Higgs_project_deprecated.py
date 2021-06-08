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






# Making another histogram, but this one takes the <120 MeV results and >130 MeV results in our range

bin_heights2, bin_edges2 = np.histogram(advanced_cut_values, range = [104, 155], bins = 30)
bin_centres2 = 0.5*(bin_edges2[1:]+bin_edges2[:-1])
bin_width2 = bin_edges2[1]-bin_edges2[0]
# Removing those two values that don't work cos they include part of the signal
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
      #
A3, A_cov= sp.optimize.curve_fit(parameter_fit, bin_centres2, bin_heights2, guess)

fit_y_values = st.get_B_expectation(bin_edges2, A3, initial_lambda)
plt.plot(bin_edges2, fit_y_values, label = 'Curvefit')
plt.legend()
# ============================================================================= USING CURVEFIT SEEMS TO WORK BEST

# [Daniel has left the chat]
# [Dillen has entered the chat]

#-----------------------------------------------------------
# part 2 (d) fitting the χ^2 method and finding the best fit
# Just to explain what is beind done with the χ^2 method:
# 1) the pull, Pi, is defined as (yi-f(θ))/σi which defines the difference between the actual data and theoretical
#    function with parameter θ.
# 2) χ^2 = ΣPi over all the individual data points
# 3) change θ and repeat
# In this case the parameter θ will be λ and A which will lopp through a bunch of different numbers.
# the measured values will be found in the array bin_heights2 and σi will be sqrt(bin_heights2)
# note: bin_heights2 is the data WITHOUT the signal data, i.e. the background exponential

# The exponential function
def exponential(lamb, A, xArray):
    return A*np.exp(-xArray/lamb)

# The pull
def pull(yArray,function,uncertaintyArray,i):
    return abs((yArray[i] - function[i])/uncertaintyArray[i])

# The χ^2 summation
def chisq(array,uncertaintyArray,lamb,A):
    temp = 0
    for i in range(len(array)):
        temp += pull(array,exponential(lamb,A,array),uncertaintyArray,i)
    return temp

# Now we can use the chi^2 method to loop through for a specific A or lambda
#first define some useful variables
uncertainty = np.sqrt(bin_heights2)

# The maximsing algorithm will contain two parts, the first of which can be ignored and replaced with a manual
# search whereas the second part is finer and will go into the detail.

# The first steps consists of taking N equally seperated steps between a lower and upper range of values, and
# run the chi^2 function. Clearly the smaller the interval the finer steps it would diescover but consequently
# the longer it takes to run. After it has run through all the steps it will then choose the value which produces
# the largest value and move to step 2. As mentioned previously this step can be removed and replaced with a
# single value

# The second step involves taking the point and going up or down and finding the maximum value.

# step (1)
# define the range to go over
valuesForAOptimise = np.linspace(0,100000,10000)
valuesForLambdaOptimise = np.linspace(1,60,20)

listOfAs = np.ndarray(len(valuesForAOptimise))
listOfLambdas = np.ndarray(len(valuesForLambdaOptimise))

# first find the maximum a using a completely arbitray lambda
for a in valuesForAOptimise:
    np.append(listOfAs,chisq(bin_heights2,uncertainty,10,a))
    print(chisq(bin_heights2,uncertainty,1,a))
# then take the index of the largest value and use that as a guess for finding lambda
roughAIndex = np.where(listOfAs == np.amin(listOfAs))[0][0]

for l in valuesForLambdaOptimise:
    np.append(listOfLambdas,chisq(bin_heights2,uncertainty,l,valuesForAOptimise[roughAIndex]))

# now we take the index of the best lambda value
roughLIndex = np.where(listOfLambdas == np.amin(listOfLambdas))[0][0]

# now we have the two rough numbers, time to optimise using an incremental method
# This method also assumes a single peak with one stationary point, the peak.

#take the largest integer,which is the previous method, as base line and take only 1/10 the size
aStep = 0.1*(valuesForAOptimise[1]-valuesForAOptimise[0])
lStep = 0.1*(valuesForLambdaOptimise[1]-valuesForLambdaOptimise[0])

def upOrDown(array,index,step):
    up = chisq(bin_heights2,uncertainty,l,array[index]+step)
    down = chisq(bin_heights2,uncertainty,l,array[index]-step)
    if up > down:
        return '+'
    elif up < down:
        return '-'
    else:
        return '='


threshold = 1e-20 #defines how accurate we need to be

print(roughAIndex)

if upOrDown(valuesForAOptimise,roughAIndex,aStep) == '+':
    while aStep > threshold:
        newA = valuesForAOptimise[roughAIndex]+aStep
        nextStep = chisq(bin_heights2,uncertainty,l,newA)
        if nextStep < chisq(bin_heights2,uncertainty,l,valuesForAOptimise[roughAIndex]):
            aStep += aStep
        else:
            newA = valuesForAOptimise[roughAIndex]-aStep
            nextStep = chisq(bin_heights2,uncertainty,l,newA)
            aStep = aStep+0.1*aStep
elif upOrDown(valuesForAOptimise,roughAIndex,aStep) == '-':
    while aStep > threshold:
        newA = valuesForAOptimise[roughAIndex]-aStep
        nextStep = chisq(bin_heights2,uncertainty,l,newA)
        if nextStep < chisq(bin_heights2,uncertainty,l,valuesForAOptimise[roughAIndex]):
            aStep += aStep
        else:
            newA = valuesForAOptimise[roughAIndex]+aStep
            nextStep = chisq(bin_heights2,uncertainty,l,newA)
            aStep *= 0.1
else:
    print('Unlikely equal error, try with differnt initial conditions')



print(newA)







plt.show()
