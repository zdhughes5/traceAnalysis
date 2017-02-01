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

print('Parsing configuration file...')
config = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
config.read(configFile)

###Initialize some variables

workingDir = Path(config['General']['workingDir'])
channel1 = workingDir/config['General']['channel1']

horizontalWidth = float(config['General']['horizontalWidth'])
horizontalGridNumber = float(config['General']['horizontalGridNumber'])
horizontalSampleNumber = float(config['General']['horizontalSampleNumber'])
verticalDivison = float(config['General']['verticalDivison'])
vertialGridNumber = float(config['General']['vertialGridNumber'])
SIL = float(config['General']['signalIntegrationLower'])
SIU = float(config['General']['signalIntegrationUpper'])
PIL = float(config['General']['pedestalIntegrationLower'])
PIU = float(config['General']['pedestalIntegrationUpper'])
voltageThreshold = float(config['General']['voltageThreshold'])
timeThreshold = int(config['General']['timeThreshold'])
saveFolder = int(config['General']['saveFolder'])

horizontalConversionPhys = horizontalWidth/horizontalGridNumber #Turns grid points/ticks into physicsal marks
horizontalConversionImag = horizontalSampleNumber/horizontalWidth #Turns physical units into array index 

horizontalSymmetricStart = horizontalGridNumber/2 - horizontalGridNumber
horizontalSymmetricStop = horizontalGridNumber/2
horizontalStart = horizontalSymmetricStart*horizontalConversionPhys
horizontalStop = horizontalSymmetricStop*horizontalConversionPhys

iSIL = (SIL+horizontalSymmetricStop)*horizontalConversionImag
iSIU = (SIU+horizontalSymmetricStop)*horizontalConversionImag
iPIL = (PIL+horizontalSymmetricStop)*horizontalConversionImag
iPIU = (PIU+horizontalSymmetricStop)*horizontalConversionImag

signalLimits = [iSIL,iSIU]
pedestalLimits = [iPIL,iPIU]

veritcalDivisionStart = 2.
verticalEnd = veritcalDivisionStart*verticalDivison
veritcalStart = -1*(vertialGridNumber-veritcalDivisionStart)*verticalDivison

############################

os.chdir(str(workingDir))

print('Extracting Traces...')
te = traceHandler.traceExtractor(config)

channel1 = te.openRawTraceFile(str(channel1))

if config['General']['saveData'] == 'True':
	subprocess.call('mkdir data',shell=True)	
	os.chdir('data')
	traceList = te.extractRawTrace(channel1,saveTraces=True,traceFilename=config['General']['savePrefix'])
	os.chdir(str(workingDir))
else:
	traceList = te.extractRawTrace(channel1)
	
subprocess.call('mkdir '+saveFolder,shell=True)
os.chdir(saveFolder)
																		
signalTracesRaw = te.extractSubtraces(traceList, signalLimits, invert = True)
pedestalTraces = te.extractSubtraces(traceList, pedestalLimits, invert = True)

pedestalSums = te.sumTraces(pedestalTraces)
offsets = -1*te.getAvgTraceMedian(pedestalSums,1000)

#for i,element in enumerate(traceList):
#	traceList[i]  = traceList[i] - np.median(traceList[i])

traceListCorrected = te.pedestalSubtractions(traceList,offsets)
signalTracesNI = te.extractSubtraces(traceListCorrected, signalLimits)
signalTracesFull = te.extractSubtraces(traceListCorrected, signalLimits, invert = True)

signalSumsFull = te.sumTraces(signalTracesFull) 

#myFont = {'fontname':'Liberation Serif'}
#plt.figure(figsize=(9,6), dpi=100)
#bins = np.linspace(np.min(pedestalSums),np.max(pedestalSums),200)
#plt.title('Summed Pedestal Distribution (Interval: ['+str(PIL)+' $\mu s$, '+str(PIU)+'$\mu s$])',**myFont)
#plt.ylabel('Number of Events [$N$]',**myFont)
#plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
#plt.hist(pedestalSums, bins, alpha=1., label='CsI Trace',color='blue')
#plt.legend(loc='upper right')
#plt.savefig('1_Summed Pedestal Distribution.png',dpi=500)
#plt.show()
#lt.close()

myFont = {'fontname':'Liberation Serif'}
plt.figure(figsize=(9,6), dpi=100)
bins = np.linspace(-1,1,250)
plt.title('Pedestal subtracted PHD for SPE (Interval: ALL)',**myFont)
plt.ylabel('Number of Events [$N$]',**myFont)
plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
n, trash, bars = plt.hist(signalSumsFull, bins, alpha=1.0, label='CsI Trace',color='blue')
plt.ylim(0,max(n)+10)
plt.legend(loc='upper right')
plt.savefig('1_Less_Simple_Summed_PHD_Full.png',dpi=500)
plt.show()
plt.close()

for i, element in enumerate(traceList):
	myFont = {'fontname':'Liberation Serif'}
	plt.figure(figsize=(9,6), dpi=100)
	plt.subplot(111)
	x = np.linspace(horizontalStart,horizontalStop,horizontalSampleNumber)
	y1 = element
	plt.plot(x, y1, color='blue', linewidth=0.5, linestyle="-")	
	plt.xticks(np.linspace(horizontalSymmetricStart,horizontalSymmetricStop,horizontalGridNumber+1,endpoint=True),**myFont)
	plt.ylim(veritcalStart,verticalEnd)
	plt.yticks(np.linspace(veritcalStart,verticalEnd,vertialGridNumber+1,endpoint=True),**myFont)
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
	#plt.show
	plt.close()
	print('beep')



























