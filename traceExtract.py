"""
Created on Tue Jan 24 17:07:06 2017

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

#My janky debugger since I'm used to IDL's STOP command
def codePause():
	import __main__
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
code.interact(local=locals())
sys.exit('Code Break!')
traceList = te.initializeData()	

############################################################
#Ok, data preparation is over; it's time for number crunching.
############################################################


#Extract the pedestal region from the CsI and WLS Fiber traces
#Sum the pedestal region
pedestalTraces = te.extractDualSubtraces(traceList, [te.pedestalLimitsPhys, te.pedestalLimitsPhys], invert = False)
pedestalSums = te.sumTraces(pedestalTraces)
del pedestalTraces

#Invert the traces for nicer histograming.
pedestalTracesInverted = te.extractDualSubtraces(traceList, [te.pedestalLimitsPhys, te.pedestalLimitsPhys], invert = True)
pedestalSumsInverted = te.sumTraces(pedestalTracesInverted)
del pedestalTracesInverted

#Calculate the pedestal offsets and correct the traceList based on offsets
#offsets is a tuple of the channel offsets. (channel1Offset, channel2Offset)
#correct the whole traceList list. Delete the old one.
offsets = te.getDualAvgMedian(pedestalSums, [te.pedestalIntervalImag,te.pedestalIntervalImag], invert = False)
traceListCorrected = te.pedestalDualSubtractions(traceList, offsets)
del traceList, pedestalSums

#Do spike rejection of traceList based on signal region
#Extract singal region from all corrected traces, invert because these are only used in sums/histograms
traceListSpikesRej = te.spikeRejection(traceListCorrected, [te.SIL,te.SIU], te.voltageThreshold, te.timeThreshold, saveSpikes=True)

#Get signal subregions
signalTracesCor = te.extractDualSubtraces(traceListCorrected, [te.signalLimitsPhys,te.signalLimitsPhys], invert=True)
signalTracesAccpt = te.extractDualSubtraces(traceListSpikesRej[0], [te.signalLimitsPhys,te.signalLimitsPhys], invert = True)
signalTracesRej = te.extractDualSubtraces(traceListSpikesRej[1], [te.signalLimitsPhys,te.signalLimitsPhys], invert = True)

#Sum the signal regions for all, accepted, and rejected trials
signalSums = te.sumTraces(signalTracesCor)
signalSumsAccpt = te.sumTraces(signalTracesAccpt)
signalSumsRej = te.sumTraces(signalTracesRej)
print('TraceListCorrected size: '+str(sys.getsizeof(traceListCorrected)))
del signalTracesCor, signalTracesAccpt, signalTracesRej, traceListCorrected


if te.savePlots == True:
	
	os.chdir(str(te.plotsFolder))

	#Prepare for plotting!
	tp = traceHandler.tracePlotter(config)

	#Plot the pedestal distributions
	tp.pedestalDualPlot(pedestalSumsInverted, te.PIL, te.PIU, 'CsI Trace', 'WLS Fiber Trace',te.horizontalUnits, '12_Summed_Pedestal_Distribution.png')

	#Fake histograms to get parameters
	binsCsI = np.linspace(np.min(signalSums[0::2]),5*signalSums[0::2].median(),100)
	binsWLS = np.linspace(np.min(signalSums[1::2]),5*signalSums[1::2].median(),100)
	nCsI, trash1, trash2 = plt.hist(signalSums[0::2], binsCsI, alpha=1.0)
	plt.close()
	nWLS, trash1, trash2 = plt.hist(signalSums[1::2], binsWLS, alpha=1.0)
	plt.close()
	nCsI = max(nCsI) + 10
	nWLS = max(nWLS) + 10


	#Make the three types of trace histogram plots for the CsI and WLS (6 total).
	tp.plotPHD(signalSums[0::2], te.SIL, te.SIU, binsCsI, 'Uncorrected CsI PHD', 'CsI Trace Sums', 
		te.horizontalUnits, 'blue', nCsI, '1_Summed_PHDPandas_Full.png')
	tp.plotPHD(signalSums[1::2], te.SIL, te.SIU, binsWLS, 'Uncorrected WSL Fiber PHD', 'WLS Fiber Trace Sums',
		te.horizontalUnits, 'red', nWLS, '2_Summed_PHDPandas_Full.png')

	tp.plotPHD(signalSumsAccpt[0::2], te.SIL, te.SIU, binsCsI, 'Corrected CsI PHD', 'CsI Trace Sums',
		te.horizontalUnits, 'blue', nCsI, '1_Summed_PHDPandas_Accepted.png')
	tp.plotPHD(signalSumsAccpt[1::2], te.SIL, te.SIU, binsWLS, 'Corrected WSL Fiber PHD', 'WLS Fiber Trace Sums',
		te.horizontalUnits, 'red', nWLS, '2_Summed_PHDPandas_Accepted.png')

	tp.plotPHD(signalSumsRej[0::2], te.SIL, te.SIU, binsCsI, 'Rejected CsI PHD', 'CsI Trace Sums',
		te.horizontalUnits, 'blue', nCsI, '1_Summed_PHDPandas_Rejected.png')
	tp.plotPHD(signalSumsRej[1::2], te.SIL, te.SIU, binsWLS, 'Rejected WSL Fiber PHD', 'WLS Fiber Trace Sums',
		te.horizontalUnits, 'red', nWLS, '2_Summed_PHDPandas_Rejected.png')

		
	for i, element in enumerate(traceListSpikesRej[0].columns[0::2]):
	
		if (i in [0,1,2,3,4,5,6,7,8,9]) or ((i % 100) == 0):
		
			tp.plotDualTrace(traceListSpikesRej[0][traceListSpikesRej[0][traceListSpikesRej[0].columns[0::2]].columns[i]],traceListSpikesRej[0][traceListSpikesRej[0][traceListSpikesRej[0].columns[1::2]].columns[i]], te.horizontalSymmetricStartPhys, 
				te.horizontalSymmetricStopPhys, te.horizontalSampleNumber, te.horizontalGridNumber,
				te.veritcalStart, te.verticalEnd, te.verticalGridNumber, 'APT Raw Detector Trace',
				te.horizontalUnits, te.SIL, te.SIU, te.PIL, te.PIU, 'blue', 'red', 'CsI PMT', 'WLS Fiber PMT',
				'APT_raw_tracePandas_'+str(i)+'.png')
			
os.chdir(str(te.workingDir))
print('All Done!')
