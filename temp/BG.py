"""
Created on Sat Jan 28 17:54:29 2017

@author: zdhughes
"""

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

def codePause():
	code.interact(local=locals())
	sys.exit('Code Break!')

garbage, configFile = argv

#Read in Configuration File
print('Parsing configuration file...')
config = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
config.read(configFile)


#Create Trace Extractor
print('Begin trace extraction...')
te = traceHandler.traceExtractor(config)
te.setupClassVariables()

print('Moving into save folder: '+te.saveFolder+'...')
os.chdir(str(te.workingDir))
subprocess.call('mkdir '+te.saveFolder,shell=True)
os.chdir(te.saveFolder)

#Read in files, return (linesChannel1, linesChannel2)
print('Reading in lines...')
lines = te.openDualRawTraceFile(te.channel1, te.channel2)
linesBG = te.openDualRawTraceFile(te.channel1BG, te.channel2BG)

#Extract the traces from the raw lines files
if te.saveTextData == 'True':
	subprocess.call('mkdir data',shell=True)	
	os.chdir('data')
	traceList = te.extractDualRawTraces(lines, saveTraces=True, traceFilename=config['General']['savePrefix'])
	traceListBG = te.extractDualRawTraces(lines, saveTraces=True, traceFilename=config['General']['savePrefix'])
	del linesBG
	del lines
	os.chdir(str(te.workingDir))
else:
	traceList = te.extractDualRawTraces(lines)
	traceListBG = te.extractDualRawTraces(lines)
	del linesBG
	del lines

#Extract the pedestal region from the CsI and WLS Fiber
#Sum the pedestal region
pedestalTraces = te.extractDualSubtraces(traceList, te.pedestalLimits, invert = False)
pedestalSums = te.sumDualTraces(pedestalTraces)
pedestalTracesBG = te.extractDualSubtraces(traceListBG, te.pedestalLimits, invert = False)
pedestalSumsBG = te.sumDualTraces(pedestalTraces)

pedestalTracesInverted = te.extractDualSubtraces(traceList, te.pedestalLimits, invert = True)
pedestalSumsInverted = te.sumDualTraces(pedestalTracesInverted)
pedestalTracesInvertedBG = te.extractDualSubtraces(traceListBG, te.pedestalLimits, invert = True)
pedestalSumsInvertedBG = te.sumDualTraces(pedestalTracesInverted)

#Calculate the pedestal offsets and correct the traceList based on offset
offsets = te.getDualAvgTraceMedian(pedestalSums, te.pedestalInterval, invert = False)
traceListCorrected = te.pedestalDualSubtractions(traceList, offsets)
offsetsBG = te.getDualAvgTraceMedian(pedestalSums, te.pedestalInterval, invert = False)
traceListCorrectedBG = te.pedestalDualSubtractions(traceListBG, offsets)
del traceListBG
del traceList

#Extract the signal region from the CsI and WLS Fiber, used in plotting and spike rejection
signalTracesRaw = te.extractDualSubtraces(traceListCorrected, te.signalLimits, invert = False)
signalTracesRawBG = te.extractDualSubtraces(traceListCorrected, te.signalLimits, invert = False)


#extract singal region from all corrected traces, invert because these are only used in sums
signalTracesCor = te.extractDualSubtraces(traceListCorrected, te.signalLimits, invert = True)
signalTracesCorBG = te.extractDualSubtraces(traceListCorrected, te.signalLimits, invert = True)


#Do spike rejection of traceList based on signal region
#Extract the signal from the accepted and rejected traces, invert because these are only used in sums
traceListSpikesRej = te.spikeRejection(traceListCorrected, signalTracesRaw, te.voltageThreshold, te.timeThreshold, saveSpikes=True)
signalTracesAccpt = te.extractDualSubtraces(traceListSpikesRej[0], te.signalLimits, invert = True)
signalTracesRej = te.extractDualSubtraces(traceListSpikesRej[1], te.signalLimits, invert = True)
traceListSpikesRejBG = te.spikeRejection(traceListCorrected, signalTracesRaw, te.voltageThreshold, te.timeThreshold, saveSpikes=True)
signalTracesAccptBG = te.extractDualSubtraces(traceListSpikesRej[0], te.signalLimits, invert = True)
signalTracesRejBG = te.extractDualSubtraces(traceListSpikesRej[1], te.signalLimits, invert = True)


#Sum the signal regions for all, accepted, and rejected trials
signalSums = te.sumDualTraces(signalTracesCor)
signalSumsBG = te.sumDualTraces(signalTracesCor)

signalSumsAccpt = te.sumDualTraces(signalTracesAccpt)
signalSumsAccptBG = te.sumDualTraces(signalTracesAccpt)

signalSumsRej = te.sumDualTraces(signalTracesRej)
signalSumsRejBG = te.sumDualTraces(signalTracesRej)

#CsI subtractions
CsISignal = signalSums[:,0]
CsIBG = signalSumsBG[:,0]

binMin = np.min(np.array((np.min(CsISignal),np.min(CsIBG))))
binMax = np.max(np.array((3*np.median(CsISignal),3*np.median(CsIBG))))

bins = np.linspace(binMin,binMax,100)

yCsISig, x = np.histogram(CsISignal, bins)
x = [(b+x[i+1])/2.0 for i,b in enumerate(x[0:-1])]
					

tp = traceHandler.tracePlotter(config)



#Make fake histograms to get bin information
binsCsI = np.linspace(np.array((np.min(signalSums[:,0]),np.min(sig))),3*np.median(signalSums[:,0]),100)
binsWLS = np.linspace(np.min(signalSums[:,1]),3*np.median(signalSums[:,1]),100)
nCsI, trash, bars = plt.hist(signalSums[:,0], binsCsI, alpha=1.0)
plt.close()
nWLS, trash, bars = plt.hist(signalSums[:,1], binsWLS, alpha=1.0)
plt.close()
nCsI = max(nCsI) + 10
nWLS = max(nWLS) + 10

#Make the three types of trace histogram plots
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

for i, element in enumerate(traceListSpikesRej[1]):
	
	tp.plotDualTrace(element[:,0], element[:,1], te.horizontalSymmetricStart, 
		te.horizontalSymmetricStop, te.horizontalSampleNumber, te.horizontalGridNumber,
		te.veritcalStart, te.verticalEnd, te.verticalGridNumber, 'APT Raw Detector Trace',
		te.horizontalUnits, te.SIL, te.SIU, te.PIL, te.PIU, 'blue', 'red', 'CsI PMT', 'WLS Fiber PMT',
		'APT_raw_trace_'+str(i)+'.png')
