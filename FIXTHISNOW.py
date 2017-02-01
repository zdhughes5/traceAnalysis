"""
Created on Fri Jan 27 20:42:53 2017

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
import traceHandler2

def codePause():
	code.interact(local=locals())
	sys.exit('Code Break!')

garbage, configFile = argv

print('Parsing configuration file...')
config = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
config.read(configFile)

###Initialize some variables

workingDir = Path(config['General']['workingDir'])
channel1 = workingDir/config['General']['channel1']
channel2 = workingDir/config['General']['channel2']

horizontalWidth = float(config['General']['horizontalWidth'])
horizontalSampleNumber = float(config['General']['horizontalSampleNumber'])
horizontalGridNumber = float(config['General']['horizontalGridNumber'])
horizontalSymmetricStart = horizontalWidth/2 - horizontalWidth
horizontalSymmetricStop = horizontalWidth/2

verticalDivison = float(config['General']['verticalDivison'])
verticalGridNumber = float(config['General']['verticalGridNumber'])
veritcalDivisionStart = 2.
verticalEnd = veritcalDivisionStart*verticalDivison
veritcalStart = -1*(verticalGridNumber-veritcalDivisionStart)*verticalDivison

SIL = float(config['General']['signalIntegrationLower'])
SIU = float(config['General']['signalIntegrationUpper'])
PIL = float(config['General']['pedestalIntegrationLower'])
PIU = float(config['General']['pedestalIntegrationUpper'])

voltageThreshold = float(config['General']['voltageThreshold'])
timeThreshold = int(config['General']['timeThreshold'])

############################



os.chdir(str(workingDir))

print('Extracting Traces...')
te = traceHandler2.traceExtractor(config)

if config['General']['saveData'] == 'True':
	subprocess.call('mkdir data',shell=True)	
	os.chdir('data')
	traceList = te.extractDualRawTraces(channel1,channel2,saveTraces=True,traceFilename=config['General']['savePrefix'])
	os.chdir(str(workingDir))
else:
	traceList = te.extractDualRawTraces(channel1,channel2)
	
subprocess.call('mkdir plots',shell=True)
os.chdir('plots')

##intsert1	##		

signalLimits = [int(1000*(SIL-horizontalSymmetricStart)),int(1000*(SIU-horizontalSymmetricStart))]
pedestalLimits = [int(1000*(PIL-horizontalSymmetricStart)),int(1000*(PIU-horizontalSymmetricStart))]
signalTracesRaw = te.extractDualSubtraces(traceList, signalLimits, invert = True)
pedestalTraces = te.extractDualSubtraces(traceList, pedestalLimits, invert = True)
pedestalSums = te.sumDualTraces(pedestalTraces)
offsets = (-1*te.getDualAvgTraceMedian(pedestalSums,1000)[0],-1*te.getDualAvgTraceMedian(pedestalSums,1000)[1])
traceListCorrected = te.pedestalDualSubtractions(traceList,offsets)
signalTracesNI = te.extractDualSubtraces(traceListCorrected, signalLimits)
traceListSpikesRej = te.spikeRejection(traceListCorrected, signalTracesNI, voltageThreshold, timeThreshold, saveSpikes=True)
signalTracesFull = te.extractDualSubtraces(traceListCorrected, signalLimits, invert = True)
signalTraces = te.extractDualSubtraces(traceListSpikesRej[0], signalLimits, invert = True)
signalTracesRej = te.extractDualSubtraces(traceListSpikesRej[1], signalLimits, invert = True)
signalSums = te.sumDualTraces(signalTraces)
signalSumsFull = te.sumDualTraces(signalTracesFull) 
signalSumsRej = te.sumDualTraces(signalTracesRej)

myFont = {'fontname':'Liberation Serif'}
plt.figure(figsize=(9,6), dpi=100)
bins = np.linspace(0,5,200)
plt.title('Summed Pedestal Distribution (Interval: ['+str(PIL)+' $\mu s$, '+str(PIU)+'$\mu s$])',**myFont)
plt.ylabel('Number of Events [$N$]',**myFont)
plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
plt.hist(pedestalSums[:,0], bins, alpha=1., label='CsI Trace',color='blue')
plt.hist(pedestalSums[:,1], bins, alpha=1., label='WLS Fiber Trace',color='red')
plt.legend(loc='upper right')
plt.savefig('1_Summed Pedestal Distribution.png',dpi=500)
plt.show()
plt.close()


############FULL


myFont = {'fontname':'Liberation Serif'}
plt.figure(figsize=(9,6), dpi=100)
bins = np.linspace(np.min(signalSumsFull[:,0]),np.max(signalSumsFull[:,0]),100)
plt.title('Less Simple Summed PHD (Interval: ['+str(SIL)+' $\mu s$, '+str(SIU)+'$\mu s$])',**myFont)
plt.ylabel('Number of Events [$N$]',**myFont)
plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
n, trash, bars = plt.hist(signalSumsFull[:,0], bins, alpha=1.0, label='CsI Trace',color='blue')
plt.ylim(0,max(n)+10)
plt.legend(loc='upper right')
plt.savefig('1_Less_Simple_Summed_PHD_Full.png',dpi=500)
plt.show()
plt.close()
	

myFont = {'fontname':'Liberation Serif'}
plt.figure(figsize=(9,6), dpi=100)
bins = np.linspace(np.min(signalSumsFull[:,1]),np.max(signalSumsFull[:,1]),100)
plt.title('Less Simple Summed PHD (Interval: ['+str(SIL)+' $\mu s$, '+str(SIU)+'$\mu s$])',**myFont)
plt.ylabel('Number of Events [$N$]',**myFont)
plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
m, trash, bars = plt.hist(signalSumsFull[:,1], bins, alpha=1.0, label='WLS Fiber Trace',color='red')
plt.ylim(0,max(m)+10)
plt.legend(loc='upper right')
plt.savefig('2_Less_Simple_Summed_PHD_Full.png',dpi=500)
plt.show()
plt.close()

################


############CORR

myFont = {'fontname':'Liberation Serif'}
plt.figure(figsize=(9,6), dpi=100)
bins = np.linspace(np.min(signalSumsFull[:,0]),np.max(signalSumsFull[:,0]),100)
plt.title('Less Simple Summed PHD (Interval: ['+str(SIL)+' $\mu s$, '+str(SIU)+'$\mu s$])',**myFont)
plt.ylabel('Number of Events [$N$]',**myFont)
plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
plt.hist(signalSums[:,0], bins, alpha=1.0, label='CsI Trace',color='blue')
plt.ylim(0,max(n)+10)
plt.legend(loc='upper right')
plt.savefig('1_Less_Simple_Summed_PHD_Corr.png',dpi=500)
plt.show()
plt.close()
	

myFont = {'fontname':'Liberation Serif'}
plt.figure(figsize=(9,6), dpi=100)
bins = np.linspace(np.min(signalSumsFull[:,1]),np.max(signalSumsFull[:,1]),100)
plt.title('Less Simple Summed PHD (Interval: ['+str(SIL)+' $\mu s$, '+str(SIU)+'$\mu s$])',**myFont)
plt.ylabel('Number of Events [$N$]',**myFont)
plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
plt.hist(signalSums[:,1], bins, alpha=1.0, label='WLS Fiber Trace',color='red')
plt.ylim(0,max(m)+10)
plt.legend(loc='upper right')
plt.savefig('2_Less_Simple_Summed_PHD_Corr.png',dpi=500)
plt.show()
plt.close()

################

############REJ

myFont = {'fontname':'Liberation Serif'}
plt.figure(figsize=(9,6), dpi=100)
bins = np.linspace(np.min(signalSumsFull[:,0]),np.max(signalSumsFull[:,0]),100)
plt.title('Less Simple Summed PHD (Interval: ['+str(SIL)+' $\mu s$, '+str(SIU)+'$\mu s$])',**myFont)
plt.ylabel('Number of Events [$N$]',**myFont)
plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
plt.hist(signalSumsRej[:,0], bins, alpha=1.0, label='CsI Trace',color='blue')
plt.ylim(0,max(n)+10)
plt.legend(loc='upper right')
plt.savefig('1_Less_Simple_Summed_PHD_Rej.png',dpi=500)
plt.show()
plt.close()
	

myFont = {'fontname':'Liberation Serif'}
plt.figure(figsize=(9,6), dpi=100)
bins = np.linspace(np.min(signalSumsFull[:,1]),np.max(signalSumsFull[:,1]),100)
plt.title('Less Simple Summed PHD (Interval: ['+str(SIL)+' $\mu s$, '+str(SIU)+'$\mu s$])',**myFont)
plt.ylabel('Number of Events [$N$]',**myFont)
plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
plt.hist(signalSumsRej[:,1], bins, alpha=1.0, label='WLS Fiber Trace',color='red')
plt.ylim(0,max(m)+10)
plt.legend(loc='upper right')
plt.savefig('2_Less_Simple_Summed_PHD_Rej.png',dpi=500)
plt.show()
plt.close()

################

############REJ


for i, element in enumerate(traceListSpikesRej[0]):
	myFont = {'fontname':'Liberation Serif'}
	plt.figure(figsize=(9,6), dpi=100)
	plt.subplot(111)
	x = np.linspace(horizontalSymmetricStart,horizontalSymmetricStop,horizontalSampleNumber)
	y1 = element[:,0]
	y2 = element[:,1]
	plt.plot(x, y1, color='blue', linewidth=0.5, linestyle="-")	
	plt.plot(x, y2, color="red", linewidth=0.5, linestyle="-")
	plt.xticks(np.linspace(horizontalSymmetricStart,horizontalSymmetricStop,horizontalGridNumber+1,endpoint=True),**myFont)
	plt.ylim(veritcalStart,verticalEnd)
	plt.yticks(np.linspace(veritcalStart,verticalEnd,verticalGridNumber+1,endpoint=True),**myFont)
	plt.grid(True)
	red_patch = mpatches.Patch(color='red', label='WLS Fiber PMT')
	blue_patch = mpatches.Patch(color='blue', label='CsI PMT')
	mpl.rc('font',family='Liberation Serif')
	plt.legend(loc='lower right',handles=[red_patch,blue_patch])
	plt.title('APT Raw Detector Trace',**myFont)
	plt.xlabel('Time Relative to Trigger [$\mu s$]',**myFont)
	plt.ylabel('Voltage [$V$]',**myFont)
	plt.plot([SIL,SIL],[veritcalStart,verticalEnd],color='black',linestyle="--",alpha=0.65)
	plt.plot([SIU,SIU],[veritcalStart,verticalEnd],color='black',linestyle="--",alpha=0.65)
	plt.plot([PIL,PIL],[veritcalStart,verticalEnd],color='grey',linestyle="--",alpha=0.65)
	plt.plot([PIU,PIU],[veritcalStart,verticalEnd],color='grey',linestyle="--",alpha=0.65)
	plt.savefig('APT_raw_trace_'+str(i)+'.png',dpi=500)
	plt.close()

################

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	