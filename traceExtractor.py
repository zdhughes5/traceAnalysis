#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:06:11 2017

@author: zdhughes
"""

import os
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
from sys import argv
import code
import sys
import traceHandler
import subprocess
	
def func(x, a, b, c):
	return a*x**2 + b*x + c

garbage, configFile = argv

print('Reading configuration file...')
config = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
config.read(configFile)

c = traceHandler.colors()
if config['General'].getboolean('ansiColors') == True:
	c.enableColors()
c.confirmColorsDonger()

print('Setting up variables...')
te = traceHandler.traceExtractor(config, c)
te.setupClassVariables()

if te.doPhotonCounting == True:
	print('Graphing photon counts...')
	binsY = np.arange(0, 60)
	te.plotPhotons(te.photonFiles, bins=binsY, labels=te.photonLabels, title=te.photonTitle,
		filename=str(te.workingDir/'allPhotons.png'), show=te.showPlots, save=te.savePlots)

else:
	print('Begin trace extraction...')
	traceList = te.initializeData()

	############################################################
	#Ok, data preparation is over; it's time for number crunching.
	############################################################

	#Extract the pedestal region from the CsI and WLS Fiber traces
	#Sum the pedestal region
	pedestalTraces = te.extractDualSubtraces(traceList, [te.windowParametersX['pedestalLimits'], te.windowParametersX['pedestalLimits']], invert = False)
	pedestalSums = te.sumTraces(pedestalTraces)
	#pedestalSumsAbs = te.sumTraces(abs(pedestalTraces))
	del pedestalTraces

	#Calculate the pedestal offsets and correct the traceList based on offsets
	#offsets is a tuple of the channel offsets. (channel1Offset, channel2Offset)
	#correct the whole traceList list. Delete the old one.
	offsets = te.getDualAvgMedian(pedestalSums, [te.dataParametersX['pedestalInterval'],te.dataParametersX['pedestalInterval']], invert = False)
	traceListCorrected = te.pedestalDualSubtractions(traceList, offsets)
	del traceList

	signalTracesCor = te.extractDualSubtraces(traceListCorrected, [te.windowParametersX['signalLimits'],te.windowParametersX['signalLimits']], invert=True)
	signalSums = te.sumTraces(signalTracesCor)
	CsITraces = traceListCorrected[traceListCorrected.columns[0::2]]
	WLSTraces = traceListCorrected[traceListCorrected.columns[1::2]]
	CsISums = signalSums[0::2].copy(deep=True)
	WLSSums = signalSums[1::2].copy(deep=True)
	del signalTracesCor, signalSums

	if te.doDoubleRejection == True:
		print('doDoubleRejection set to '+c.lightblue(str(te.doDoubleRejection))+'. Rejecting doubles...')
		acceptedTraces, rejectedTraces, good, bad = te.doubleRejection(CsITraces, te.windowParametersX,
			te.dataParametersX, te.SGWindow, te.SGOrder, te.minimaWindowDR, te.medianFactorDR,
			te.fitWindow, te.alphaThreshold)
		CsISumsCulled = CsISums.drop(CsISums.index[bad])
		WLSSumsCulled = WLSSums.drop(WLSSums.index[bad])	
	
	if te.doPeakFinder == True:
		print('doPeakFinder set to '+c.lightblue(str(te.doPeakFinder))+'. Finding peaks...')
		if te.doDoubleRejection == True: 
			photonInd, countedPhotons = te.peakFinder(WLSTraces.drop(WLSTraces[bad].columns,axis=1), te.windowParametersX, te.dataParametersX,
				te.medianFactorPF, te.stdFactor, te.convWindow, te.convPower, te.convSig, te.minimaWindowPF)
		else:
			photonInd, countedPhotons = te.peakFinder(WLSTraces, te.windowParametersX, te.dataParametersX,
				te.medianFactorPF, te.stdFactor, te.convWindow, te.convPower, te.convSig, te.minimaWindowPF)		
		subprocess.call('mkdir -p '+str(te.plotsDir), shell=True)
		subprocess.call('mkdir -p '+str(te.traceDir), shell=True)
		f = open(te.photonFilename,'w')
		for i, ele in enumerate(countedPhotons):
			f.write(str(ele)+'\n')
		f.close()
	


	if te.doPlots == True:
	
		print('doPlots set to '+c.lightblue(str(te.doPlots))+'. Plotting...')
		subprocess.call('mkdir -p '+str(te.plotsDir), shell=True)
		subprocess.call('mkdir -p '+str(te.traceDir), shell=True)

		#Prepare for plotting!
		#tp = traceHandler.tracePlotter(config)
	
		#Plot the pedestal distributions
		print('Plotting pedestals...')
		te.pedestalPlot(pedestalSums[0::2], te.windowParametersX, te.windowParametersY1, fileName=str(te.plotsDir/'00_CsI_pedestal.png'), show=te.showPlots, save=te.savePlots)
		te.pedestalPlot(pedestalSums[1::2], te.windowParametersX, te.windowParametersY2, fileName=str(te.plotsDir/'00_WLS_pedestal.png'), show=te.showPlots, save=te.savePlots)
	
		#Make the three types of trace histogram plots for the CsI and WLS (6 total).
		print('Plotting PHDs...')
		binsCsI = te.plotPHD(CsISums, te.windowParametersX, te.windowParametersY1, fileName=str(te.plotsDir/'01_CsI_PHD_Full.png'), show=te.showPlots, save=te.savePlots)
		binsWLS = te.plotPHD(WLSSums, te.windowParametersX, te.windowParametersY2, fileName=str(te.plotsDir/'01_WLS_PHD_Full.png'), show=te.showPlots, save=te.savePlots)
	
		if te.doDoubleRejection == True:
			te.plotPHD(CsISumsCulled, te.windowParametersX, te.windowParametersY1, bins=binsCsI, fileName=str(te.plotsDir/'01_CsI_PHD_Culled.png'), show=te.showPlots, save=te.savePlots)
			te.plotPHD(WLSSumsCulled, te.windowParametersX, te.windowParametersY2, bins=binsWLS, fileName=str(te.plotsDir/'01_WLS_PHD_Culled.png'), show=te.showPlots, save=te.savePlots)	
		
		if te.doPeakFinder == True:
			binsY = np.arange(0, 60)
			te.plotPHD(countedPhotons, te.windowParametersX, te.windowParametersY2, legend='WLS Photons',
				color='cyan', yLabel='Counted photons [$N$]', title='Counted photon distribution', bins=binsY, ylim=500,fileName=str(te.plotsDir/'02_photon_PHD.png'), show=te.showPlots, save=te.savePlots)

		#code.interact(local=locals())
		#sys.exit('Code Break!')	
	
		if te.doDoubleRejection == True:
			print('Plotting traces...')
			if te.allPlots == True:	
				for i, index in enumerate(bad):
					filename = 'rejected_trace_'+str(index)+'.png'
					te.plotDualTrace(CsITraces[CsITraces.columns[index]], WLSTraces[WLSTraces.columns[index]], te.windowParametersX,
						te.windowParametersY1, te.windowParametersY2, title='Trace '+str(index), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)
					if i == 0:
						te.plotDualTrace(CsITraces[CsITraces.columns[index]], WLSTraces[WLSTraces.columns[index]], te.windowParametersX,
							te.windowParametersY1, te.windowParametersY2, title='Trace '+str(index), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)			
			else:
				for i, index in enumerate(bad):
					if (i in [0,1,2,3,4,5,6,7,8,9]) or ((i % 100) == 0):
						filename = 'rejected_trace_'+str(index)+'.png'
						te.plotDualTrace(CsITraces[CsITraces.columns[index]], WLSTraces[WLSTraces.columns[index]], te.windowParametersX,
							te.windowParametersY1, te.windowParametersY2, title='Trace '+str(index), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)
						if i == 0:
							te.plotDualTrace(CsITraces[CsITraces.columns[index]], WLSTraces[WLSTraces.columns[index]], te.windowParametersX,
								te.windowParametersY1, te.windowParametersY2, title='Trace '+str(index), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)							

			if te.allPlots == True:	
				for i, index in enumerate(good):
					filename = 'accepted_trace_'+str(index)+'.png'
					te.plotDualTrace(CsITraces[CsITraces.columns[index]], WLSTraces[WLSTraces.columns[index]], te.windowParametersX,
						te.windowParametersY1, te.windowParametersY2, title='Trace '+str(index), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)
					if i == 0:
						te.plotDualTrace(CsITraces[CsITraces.columns[index]], WLSTraces[WLSTraces.columns[index]], te.windowParametersX,
							te.windowParametersY1, te.windowParametersY2, title='Trace '+str(index), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)			
			else:
				for i, index in enumerate(good):
					if (i in [0,1,2,3,4,5,6,7,8,9]) or ((i % 100) == 0):
						filename = 'accepted_trace_'+str(index)+'.png'
						te.plotDualTrace(CsITraces[CsITraces.columns[index]], WLSTraces[WLSTraces.columns[index]], te.windowParametersX,
							te.windowParametersY1, te.windowParametersY2, title='Trace '+str(index), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)
						if i == 0:
							te.plotDualTrace(CsITraces[CsITraces.columns[index]], WLSTraces[WLSTraces.columns[index]], te.windowParametersX,
								te.windowParametersY1, te.windowParametersY2, title='Trace '+str(index), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)							
		else:
			print('Plotting traces...')
			if te.allPlots == True:
				for i, index in enumerate(CsITraces.columns):
					filename = 'trace_'+str(i)+'.png'
					te.plotDualTrace(CsITraces[index], WLSTraces[index], te.windowParametersX, te.windowParametersY1,
						te.windowParametersY2, title='Trace '+str(i), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)
					if i == 0:
						te.plotDualTrace(CsITraces[index], WLSTraces[index], te.windowParametersX, te.windowParametersY1,
							te.windowParametersY2, title='Trace '+str(i), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)		
			else:
				for i, index in enumerate(CsITraces.columns):
					if (i in [0,1,2,3,4,5,6,7,8,9]) or ((i % 100) == 0):
						filename = 'trace_'+str(i)+'.png'
						te.plotDualTrace(CsITraces[CsITraces.columns[i]], WLSTraces[WLSTraces.columns[i]], te.windowParametersX, te.windowParametersY1,
							te.windowParametersY2, title='Trace '+str(i), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)
						if i == 0:
							te.plotDualTrace(CsITraces[CsITraces.columns[i]], WLSTraces[WLSTraces.columns[i]], te.windowParametersX, te.windowParametersY1,
								te.windowParametersY2, title='Trace '+str(i), fileName=str(te.traceDir/filename), show=te.showPlots, save=te.savePlots)				
				

os.chdir(str(te.workingDir))
print('All Done!\n')
#code.interact(local=locals())
#sys.exit('Code Break!')
