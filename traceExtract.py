"""
Created on Tue Jan 24 17:07:06 2017

@author: zdhughes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import tkinter 
from configparser import ConfigParser, ExtendedInterpolation
from sys import argv
import code
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl
import subprocess
from pathlib import Path
import traceHandler
import pandas as pd

def codePause():
	code.interact(local=locals())
	sys.exit('Code Break!')

garbage, configFile = argv

#Read in Configuration File. Creates config object, then reads in file.
print('Parsing configuration file...')
config = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
config.read(configFile)


#Create Trace Extractor object. Set up variables from config file.
print('Begin trace extraction...')
te = traceHandler.traceExtractor(config)
te.setupClassVariables()

#Move into directories.
print('Moving into save folder: '+te.saveFolder+'...')
os.chdir(str(te.workingDir))
subprocess.call('mkdir '+te.saveFolder,shell=True)
os.chdir(te.saveFolder)

#Read in files, return (linesChannel1, linesChannel2)
print('Reading in lines...')
lines = te.openDualRawTraceFile(te.channel1, te.channel2)

#Extract the trace data from the raw lines. Save if directed.
if te.saveTextData == 'True':
	subprocess.call('mkdir data',shell=True)	
	os.chdir('data')
	traceList = te.extractDualRawTraces(lines, saveTraces=True, traceFilename=te.savePrefix)
	os.chdir('..')
	del lines
else:
	traceList = te.extractDualRawTraces(lines)
	del lines


Interval = np.linspace(te.horizontalSymmetricStart,te.horizontalSymmetricStop,te.horizontalSampleNumber)
traceList2 = te.dualTraceToPandas(traceList, Interval)

code.interact(local=locals())
sys.exit('Code Break!')
	
#Ok, data preparation is over; it's time for number crunching.
	

#Extract the pedestal region from the CsI and WLS Fiber traces
#Sum the pedestal region
#pedestalTraces is a list of 2d numpy arrays. Shape: [# of traces][# of samples, # of channels]
#pedestalSums is a 2d numpy array. [# of traces, # of channels]
pdLimits = [te.PIL,te.PIU]
pedestalTraces2 = te.extractDualPandasSubtraces(traceList2, pdLimits, pdLimits, invert = False)
pedestalSums2 = te.sumPandasTraces(pedestalTraces2)

pedestalTraces = te.extractDualSubtraces(traceList, te.pedestalLimits, invert = False)
pedestalSums = te.sumDualTraces(pedestalTraces)
del pedestalTraces

#Invert the traces for nicer histograming.
pedestalTracesInverted2 = te.extractDualPandasSubtraces(traceList2, pdLimits, pdLimits, invert = False)
pedestalSumsInverted2 = te.sumDualTraces(pedestalTracesInverted2)

pedestalTracesInverted = te.extractDualSubtraces(traceList, te.pedestalLimits, invert = True)
pedestalSumsInverted = te.sumDualTraces(pedestalTracesInverted)

#Calculate the pedestal offsets and correct the traceList based on 
# offsets is a tuple of the channel offsets. (channel1Offset, channel2Offset)
#correct the whole traceList list. Delete the old one.
offsets2 = te.getDualAvgTraceMedian(pedestalSums2, [te.pedestalInterval,te.pedestalInterval], invert = False)
traceListCorrected2 = te.extractDualPandasSubtraces(traceList2, offsets2)

offsets = te.getDualAvgTraceMedian(pedestalSums, te.pedestalInterval, invert = False)
traceListCorrected = te.pedestalDualSubtractions(traceList, offsets)
code.interact(local=locals())
sys.exit('Code Break!')
del traceList

#Extract the signal region from the CsI and WLS Fiber, used in plotting and spike rejection
#Shape: [# of traces][sample # in sigal region, # of channels]
signalTracesRaw = te.extractDualSubtraces(traceListCorrected, te.signalLimits, invert = False)
#extract singal region from all corrected traces, invert because these are only used in sums/histograms
signalTracesCor = te.extractDualSubtraces(traceListCorrected, te.signalLimits, invert = True)

#Do spike rejection of traceList based on signal region
#Extract the signal from the accepted and rejected traces, invert because these are only used in sums/histograms
traceListSpikesRej = te.spikeRejection(traceListCorrected, signalTracesRaw, te.voltageThreshold, te.timeThreshold, saveSpikes=True)
signalTracesAccpt = te.extractDualSubtraces(traceListSpikesRej[0], te.signalLimits, invert = True)
signalTracesRej = te.extractDualSubtraces(traceListSpikesRej[1], te.signalLimits, invert = True)

#Sum the signal regions for all, accepted, and rejected trials
signalSums = te.sumDualTraces(signalTracesCor)
signalSumsAccpt = te.sumDualTraces(signalTracesAccpt)
signalSumsRej = te.sumDualTraces(signalTracesRej)
del signalTracesCor, signalTracesAccpt, signalTracesRej

#Prepare for plotting!
tp = traceHandler.tracePlotter(config)

#Plot the pedestal distributions
tp.pedestalDualPlot(pedestalSumsInverted, te.PIL, te.PIU, 'CsI Trace', 'WLS Fiber Trace',
	te.horizontalUnits, '12_Summed_Pedestal_Distribution.png')

#Make fake histograms to get bin information and canvas dimensions
binsCsI = np.linspace(np.min(signalSums[:,0]),5*np.median(signalSums[:,0]),100)
binsWLS = np.linspace(np.min(signalSums[:,1]),5*np.median(signalSums[:,1]),100)
nCsI, trash1, trash2 = plt.hist(signalSums[:,0], binsCsI, alpha=1.0)
plt.close()
nWLS, trash1, trash2 = plt.hist(signalSums[:,1], binsWLS, alpha=1.0)
plt.close()
nCsI = max(nCsI) + 10
nWLS = max(nWLS) + 10



#Make the three types of trace histogram plots for the CsI and WLS (6 total).
tp.plotPHD(signalSums[:,0], te.SIL, te.SIU, binsCsI, 'Uncorrected CsI PHD', 'CsI Trace Sums', 
	te.horizontalUnits, 'blue', nCsI, '1_Summed_PHD_Full.png')
tp.plotPHD(signalSums[:,1], te.SIL, te.SIU, binsWLS, 'Uncorrected WSL Fiber PHD', 'WLS Fiber Trace Sums',
	te.horizontalUnits, 'red', nWLS, '2_Summed_PHD_Full.png')

tp.plotPHD(signalSumsAccpt[:,0], te.SIL, te.SIU, binsCsI, 'Corrected CsI PHD', 'CsI Trace Sums',
	te.horizontalUnits, 'blue', nCsI, '1_Summed_PHD_Accepted.png')
tp.plotPHD(signalSumsAccpt[:,1], te.SIL, te.SIU, binsWLS, 'Corrected WSL Fiber PHD', 'WLS Fiber Trace Sums',
	te.horizontalUnits, 'red', nWLS, '2_Summed_PHD_Accepted.png')

tp.plotPHD(signalSumsRej[:,0], te.SIL, te.SIU, binsCsI, 'Rejected CsI PHD', 'CsI Trace Sums',
	te.horizontalUnits, 'blue', nCsI, '1_Summed_PHD_Rejected.png')
tp.plotPHD(signalSumsRej[:,1], te.SIL, te.SIU, binsWLS, 'Rejected WSL Fiber PHD', 'WLS Fiber Trace Sums',
	te.horizontalUnits, 'red', nWLS, '2_Summed_PHD_Rejected.png')


#Print out the first 10 traces and every 100 after that.
for i, element in enumerate(traceListSpikesRej[0]):
	
	if (i in [0,1,2,3,4,5,6,7,8,9]) or ((i % 100) == 0):
		
		tp.plotDualTrace(element[:,0], element[:,1], te.horizontalSymmetricStart, 
			te.horizontalSymmetricStop, te.horizontalSampleNumber, te.horizontalGridNumber,
			te.veritcalStart, te.verticalEnd, te.verticalGridNumber, 'APT Raw Detector Trace',
			te.horizontalUnits, te.SIL, te.SIU, te.PIL, te.PIU, 'blue', 'red', 'CsI PMT', 'WLS Fiber PMT',
			'APT_raw_trace_'+str(i)+'.png')
