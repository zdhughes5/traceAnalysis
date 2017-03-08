#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:06:11 2017

@author: zdhughes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser, ExtendedInterpolation
from sys import argv
import code
import sys
import traceHandler
import pandas as pd
import subprocess
from scipy import signal
from scipy.optimize import curve_fit
from math import sqrt

#My janky debugger since I'm used to IDL's STOP command
def codePause():
	import __main__
	code.interact(local=locals())
	sys.exit('Code Break!')
	
def func(x, a, b, c):
	return a*x**2 +b*x+ c

garbage, configFile = argv

print('Reading configuration file...')
config = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
config.read(configFile)

c = traceHandler.colors()
if config['General'].getboolean('ansiColors') == True:
	c.enableColors()
	c.confirmColors()
	
print('Begin trace extraction...')
te = traceHandler.traceExtractor(config, c)
te.setupClassVariables()
traceList = te.initializeData()

############################################################
#Ok, data preparation is over; it's time for number crunching.
############################################################


#Extract the pedestal region from the CsI and WLS Fiber traces
#Sum the pedestal region
pedestalTraces = te.extractDualSubtraces(traceList, [te.windowParametersX['pedestalLimits'], te.windowParametersX['pedestalLimits']], invert = False)
pedestalSums = te.sumTraces(pedestalTraces)
#del pedestalTraces

#Invert the traces for nicer histograming.
#pedestalTracesInverted = te.extractDualSubtraces(traceList, [te.windowParametersX['pedestalLimits'], te.windowParametersX['pedestalLimits']], invert = True)
pedestalSumsAbs = te.sumTraces(abs(pedestalTraces))
#del pedestalTracesInverted

#Calculate the pedestal offsets and correct the traceList based on offsets
#offsets is a tuple of the channel offsets. (channel1Offset, channel2Offset)
#correct the whole traceList list. Delete the old one.
offsets = te.getDualAvgMedian(pedestalSums, [te.dataParametersX['pedestalInterval'],te.dataParametersX['pedestalInterval']], invert = False)
traceListCorrected = te.pedestalDualSubtractions(traceList, offsets)
#del traceList, pedestalSums

signalTracesCor = te.extractDualSubtraces(traceListCorrected, [te.windowParametersX['signalLimits'],te.windowParametersX['signalLimits']], invert=True)
signalSums = te.sumTraces(signalTracesCor)


if te.savePlots == True:
	
	subprocess.call('mkdir -p '+str(te.plotsFolder), shell=True)
	os.chdir(str(te.plotsFolder))
	subprocess.call('mkdir -p traces3', shell=True)
	os.chdir('traces3')

	#Prepare for plotting!
	tp = traceHandler.tracePlotter(config)
	
	#Plot the pedestal distributions
	tp.pedestalPlot(pedestalSumsAbs[0::2], te.windowParametersX, te.windowParametersY1, fileName='00_WLS_pedestal.png')
	tp.pedestalPlot(pedestalSumsAbs[1::2], te.windowParametersX, te.windowParametersY2, fileName='00_CsI_pedestal.png')
	
	#Make the three types of trace histogram plots for the CsI and WLS (6 total).
	tp.plotPHD(signalSums[0::2], te.windowParametersX, te.windowParametersY1, fileName='00_Full_WLS_PHD.png')
	tp.plotPHD(signalSums[1::2], te.windowParametersX, te.windowParametersY2, fileName='00__Full_CsI_PHD.png')
	newSignalSums1 = signalSums[0::2].copy(deep=True)
	newSignalSums2 = signalSums[1::2].copy(deep=True)	
	
	countedPhotons = []
	droppedSums = []
	k = 0
	for i, element in enumerate(traceListCorrected.columns[0::2]):
		
		CsITrace = np.array(traceListCorrected[traceListCorrected[traceListCorrected.columns[1::2]].columns[i]])
		CsISmoothed = signal.savgol_filter(CsITrace, 501, 1, deriv=0)
		minIndex = signal.argrelmin(CsISmoothed, order=751)[0]
		medianCutoff = abs(traceListCorrected[traceListCorrected.columns[1::2]][te.windowParametersX['PIL']:te.windowParametersX['PIU']]).median(axis=1).median()
		medianCutoff = 2*medianCutoff
		newIndex = minIndex[np.where(CsISmoothed[minIndex] < -1*medianCutoff)]
		#print('**********************************')
		numGT = []
		for j, thing in enumerate(newIndex):
			try:
				xdata = te.windowParametersX['x'][thing-250:thing+250]
			except IndexError:
				#print('Too close to edge, skipping...')
				continue
			
			ydata = CsISmoothed[thing-250:thing+250]
			popt, pcov = curve_fit(func, xdata, ydata)
			if popt[0] > 0.035:
				numGT.append(popt[0])
			
			#print('Optimized points for index '+str(thing)+' and run '+str(i)+':')
			#print(popt)
			
		#print('**********************************')
		if len(numGT) > 1:
			droppedSums.append(i)
			#plt.figure(figsize=(9,6), dpi=100)
			#plt.plot(te.windowParametersX['x'], CsITrace, alpha=0.5, color='blue')
			#plt.plot(te.windowParametersX['x'], CsISmoothed, color='red', linewidth=2.0)
			#plt.plot(te.windowParametersX['x'][newIndex], CsISmoothed[newIndex], 'r+')
			#plt.savefig('zz_rejected_'+str(i)+'.png',dpi=500)
			#plt.close()
			#code.interact(local=locals())
			#sys.exit('Code Break!')
			tp.plotDualTrace(traceListCorrected[traceListCorrected[traceListCorrected.columns[0::2]].columns[i]],
				traceListCorrected[traceListCorrected[traceListCorrected.columns[1::2]].columns[i]],
				te.windowParametersX, te.windowParametersY1, te.windowParametersY2,
				fileName='rejected_trace_'+str(i)+'.png')
		else:
			k += 1
			# = np.array(-1*traceListCorrected[traceListCorrected.columns[0::2]].ix[:,i][te.windowParametersX['SIL']:te.windowParametersX['SIU']])
			#dataInt = [0, len(data)]
			#oldData = np.copy(data)
			#medCutoff = 2*np.median(abs(data))
			#stdCutoff = 2*np.std(data[np.where(abs(data) < medCutoff)])
			#window = signal.general_gaussian(21, p=0.5, sig=10)
			#filtered = signal.fftconvolve(window, data)
			#filtered = (np.average(data) / np.average(filtered)) * filtered
			#filtered = np.roll(filtered, -10)
			#peakind = signal.argrelmax(filtered, order=11)[0]
			#newind = peakind[np.where(filtered[peakind] > stdCutoff)]
			#countedPhotons.append(len(newind))
			
			data = np.array(-1*traceListCorrected[traceListCorrected.columns[0::2]].ix[:,i][te.windowParametersX['SIL']:te.windowParametersX['SIU']])
			dataInt = [0, len(data)]
			oldData = np.copy(data)
			medCutoff = 3*np.median(abs(data))
			stdCutoff = 3*np.std(data[np.where(abs(data) < medCutoff)])
			window = signal.general_gaussian(7, p=0.5, sig=10)
			filtered = signal.fftconvolve(window, data)
			filtered = (np.average(data) / np.average(filtered)) * filtered
			filtered = np.roll(filtered, -3)
			peakind = signal.argrelmax(filtered, order=7)[0]
			newind = peakind[np.where(filtered[peakind] > stdCutoff)]
			countedPhotons.append(len(newind))			
	
			#if (k in [0,1,2,3,4,5,6,7,8,9]) or ((k % 100) == 0):
				
			tp.plotDualTrace(traceListCorrected[traceListCorrected[traceListCorrected.columns[0::2]].columns[i]],
				traceListCorrected[traceListCorrected[traceListCorrected.columns[1::2]].columns[i]],
				te.windowParametersX, te.windowParametersY1, te.windowParametersY2,
				fileName='APT_raw_trace_'+str(i)+'.png')
				
			plt.plot(filtered[0:len(data)],'blue', linewidth=0.5)
			plt.plot(newind,filtered[newind],'rx')
			plt.plot([0,len(data)],[stdCutoff,stdCutoff])
			plt.savefig('peaks_'+str(i)+'.png',dpi=500)
			plt.close()

	newSignalSums1 = newSignalSums1.drop(newSignalSums1.index[droppedSums])
	newSignalSums2 = newSignalSums2.drop(newSignalSums2.index[droppedSums])		
	binsWLS = np.linspace(np.min(signalSums[0::2]),5*signalSums[0::2].median(), int(round(2*sqrt(signalSums[0::2].index.shape[0]))))
	binsCsI = np.linspace(np.min(signalSums[1::2]),5*signalSums[1::2].median(), int(round(2*sqrt(signalSums[1::2].index.shape[0]))))
	tp.plotPHD(newSignalSums1, te.windowParametersX, te.windowParametersY1, bins=binsWLS, fileName='00_Culled_WLS_PHD.png')
	tp.plotPHD(newSignalSums2, te.windowParametersX, te.windowParametersY2, bins=binsCsI, fileName='00__Culled_CsI_PHD.png')			
				
	binsY = np.arange(np.min(countedPhotons),np.max(countedPhotons)+5)
	tp.plotPHD(countedPhotons, te.windowParametersX, te.windowParametersY1, legend='WLS Photons',
		color='cyan', yLabel='Counted photons [$N$]', title='Counted photon distribution', bins=binsY, fileName='00_photon_PHD.png')
	



			
os.chdir(str(te.workingDir))
print('All Done!')
code.interact(local=locals())
sys.exit('Code Break!')