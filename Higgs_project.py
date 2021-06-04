#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:02:44 2021

@author: samueltsang
"""

import STOM_higgs_tools as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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