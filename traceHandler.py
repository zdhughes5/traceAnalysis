"""
Created on Thu Jan 26 17:45:11 2017

@author: zdhughes
"""

import sys
import numpy as np
from pathlib import Path
import code
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def codePause():
	code.interact(local=locals())
	sys.exit('Code Break!')

class traceExtractor:
	
	def __init__(self, config = None):
		self.config = config if config is not None else sys.exit('No config found for traceExtractor. Aborting.')

		
	#Reads in values from config file and creates all forseeable variables.		
	def setupClassVariables(self):

		#Read in config variables
		self.workingDir = Path(self.config['General']['workingDir'])
		self.channel1 = self.workingDir/self.config['General']['channel1']
		self.channel2 = self.workingDir/self.config['General']['channel2']
		self.channel1BG = self.workingDir/self.config['General']['channel1BG']
		self.channel2BG = self.workingDir/self.config['General']['channel2BG']
		self.SIL = float(self.config['General']['signalIntegrationLower'])
		self.SIU = float(self.config['General']['signalIntegrationUpper'])
		self.PIL = float(self.config['General']['pedestalIntegrationLower'])
		self.PIU = float(self.config['General']['pedestalIntegrationUpper'])
		self.horizontalGridNumber = float(self.config['General']['horizontalGridNumber'])
		self.horizontalWidth = float(self.config['General']['horizontalWidth'])
		self.voltageThreshold = float(self.config['General']['voltageThreshold'])
		self.timeThreshold = int(self.config['General']['timeThreshold'])
		self.horizontalSampleNumber = float(self.config['General']['horizontalSampleNumber'])
		self.horizontalUnits = self.config['General']['horizontalUnits']
		self.verticalUnits = self.config['General']['verticalUnits']
		if self.horizontalUnits == 's':
			self.horizontalUnits = '$Seconds$'			
		elif self.horizontalUnits == 'm':
			self.horizontalUnits = '$milliseconds$'		
		elif self.horizontalUnits == 'u':
			self.horizontalUnits = '$\mu s$'
		elif self.horizontalUnits == 'n':
			self.horizontalUnits = '$ns$'
		self.verticalDivison = float(self.config['General']['verticalDivison'])
		self.verticalGridNumber  = float(self.config['General']['verticalGridNumber'])
		self.saveTextData = self.config['General']['saveTextData']
		self.savePrefix = self.config['General']['savePrefix']
		self.saveFolder = self.config['General']['saveFolder']
		
		#Calculate some values
		self.horizontalConversionPhys = self.horizontalWidth/self.horizontalGridNumber #Turns grid points/ticks into physicsal marks
		self.horizontalConversionImag = self.horizontalSampleNumber/self.horizontalWidth #Turns physical units into array index 
		self.horizontalSymmetricStart = self.horizontalWidth/2 - self.horizontalWidth
		self.horizontalSymmetricStop = self.horizontalWidth/2		
		self.horizontalStart = self.horizontalSymmetricStart*self.horizontalConversionPhys
		self.horizontalStop = self.horizontalSymmetricStop*self.horizontalConversionPhys
		self.iSIL = int((self.SIL+self.horizontalSymmetricStop)*self.horizontalConversionImag)
		self.iSIU = int((self.SIU+self.horizontalSymmetricStop)*self.horizontalConversionImag)
		self.iPIL = int((self.PIL+self.horizontalSymmetricStop)*self.horizontalConversionImag)
		self.iPIU = int((self.PIU+self.horizontalSymmetricStop)*self.horizontalConversionImag)
		self.signalLimits = [int(self.iSIL),int(self.iSIU)]
		self.pedestalLimits = [int(self.iPIL),int(self.iPIU)]
		self.signalInterval = self.iSIU - self.iSIL
		self.pedestalInterval = self.iPIU - self.iPIL
		self.veritcalDivisionStart = 2.
		self.verticalEnd = self.veritcalDivisionStart*self.verticalDivison
		self.veritcalStart = -1*(self.verticalGridNumber-self.veritcalDivisionStart)*self.verticalDivison
		
		#Print sanity check
		print('Time for a sanity check...!')
		print('Physical-to-image signal integration window map: [ '+str(self.SIL)+','+str(self.SIU)+' ] --> [ '+str(self.iSIL)+','+str(self.iSIU)+' ]')
		print('Physical-to-image pedestal integration window map: [ '+str(self.PIL)+','+str(self.PIU)+' ] --> [ '+str(self.iPIL)+','+str(self.iPIU)+' ]')
	
		
	##########	
		
	#Turns two numpy arrays column stacked 2d numpy array.
	def packageTrace(self, trace1, trace2):
		
		return np.column_stack((trace1,trace2))

		
	##########	

	#Takes in a path to oscilliscope file and returns the lines.	
	def openRawTraceFile(self, channel1File):
		with open(str(channel1File),'r') as f:
			lines = [line.strip('\n') for line in f]
													
		return lines
		
	#Convenience function that opens two oscilliscope files.
	def openDualRawTraceFile(self, channel1File, channel2File):
		
		lines1 = self.openRawTraceFile(channel1File)
		lines2 = self.openRawTraceFile(channel2File)
													
		return (lines1, lines2)
	
		
	##########	
	
	#Parses the lines of the oscilliscope file and pulls out just the trace data. Optional Save.
	def extractRawTraces(self, lines, saveTraces=False, traceFilename = 'trace_'):
		
		totalTraces = len([x for x in lines if x == 'data:'])
		traceList = []

		for i, trash  in enumerate(range(totalTraces)):

			lines[7+i*10007:10007+10007]
			extractedTrace = np.array(list(map(float,lines[7+i*10007:10007+i*10007])))
			traceList.append(extractedTrace)
			
			if saveTraces == True:
				np.savetxt(traceFilename+str(i)+'.txt',traceList[i],fmt='%.6e')
			
		return traceList
		
	#Convenience function that parse two lists of lines.
	def extractDualRawTraces(self, lines, saveTraces=False, traceFilename = 'trace_'):
		
		channel1Data = self.extractRawTraces(lines[0])
		channel2Data = self.extractRawTraces(lines[1])
		returnData = []
		
		for i, element in enumerate(channel1Data):
			returnData.append(self.packageTrace(channel1Data[i],channel2Data[i]))
			if saveTraces == True:
				np.savetext(traceFilename+str(i)+returnData[i]+'.txt',fmt='%.6e')
		
		return returnData

		
	##########	
		
	#Extracts a subset of given numpy array based on limits. Optional invert.
	def extractSubtrace(self, traceData, limits, invert= False):
		
		if invert == True:
			returnData = -1*traceData[limits[0]:limits[1]]
		else:
			returnData = traceData[limits[0]:limits[1]]
			
		return returnData
	
	#Convenience function to extract two subtraces.
	def extractDualSubtrace(self, traceData, limits, invert = False):
		
		#Takes in two single np arrays/lists and pulls out subarrays based on limits
		returnData1 = self.extractSubtrace(traceData[:,0],limits, invert = invert)
		returnData2 = self.extractSubtrace(traceData[:,1],limits, invert = invert)
		
		return self.packageTrace(returnData1,returnData2)
		
	#Convenience function that extracts subset from list of numpy arrays.
	def extractSubtraces(self, traceData, limits, invert = False):
		
		returnData = []
		
		for i, element in enumerate(traceData):
			if invert == False:
				returnData.append(self.extractSubtrace(element, limits))
			elif invert == True:
				returnData.append(-1*self.extractSubtrace(element, limits))
		
		return returnData
		
	#Convenience function for list that contains 2d numpy arrays.
	def extractDualSubtraces(self, traceData, limits, invert = False):
		
		#Takes in a list of np column stacks and pulls out subarrays based on limits
		returnData = []
		#Iterate over list length
		for i, element in enumerate(traceData):
			returnData.append(self.extractDualSubtrace(element, limits, invert = invert))
			
		return returnData

		
	##########	
	
	#Sum an array
	def sumTrace(self, traceData):
		
		return np.sum(traceData)
	
	#Convenience function for two arrays.
	def sumDualTrace(self, traceData1, traceData2):
		
		returnData1 = self.sumTrace(traceData1)
		returnData2 = self.sumTrace(traceData2)
		
		return self.packageTrace(returnData1,returnData2)

	#Convenience function for a list of arrays.
	def sumTraces(self, traceData):
		
		returnData = []
		
		for i, element in enumerate(traceData):
			returnData.append(np.sum(element))
			
		return returnData
	
	#Convenience function for list of 2d arrays.
	def sumDualTraces(self, traceData):
		
		#a list of np column stacks
		returnData1 = []
		returnData2 = []
		
		for i, element in enumerate(traceData):
			returnData1.append(self.sumTrace(element[:,0]))
			returnData2.append(self.sumTrace(element[:,1]))
			
		returnData1 = np.array(returnData1)
		returnData2 = np.array(returnData2)
					
		return self.packageTrace(returnData1, returnData2)
	
		
	##########	
	
	#Get median of an array.
	def getTraceMedian(self, traceData):
		
		return np.median(traceData)
	
	#Get average of median value in a list.			
	def getAvgTraceMedian(self, traceData, intervalSize, invert = False):

		if invert  == True:
			traceMedian = self.getTraceMedian(traceData)
			return -1*(traceMedian/intervalSize)
		elif invert  == False:
			traceMedian = self.getTraceMedian(traceData)
			return traceMedian/intervalSize
	
	#Convenience function for 2d array
	def getDualAvgTraceMedian(self, traceData, intervalSize, invert = False):
		
		#traceData = np column stack of summed values
		traceResult1 = self.getAvgTraceMedian(traceData[:,0], intervalSize, invert = invert)
		traceResult2 = self.getAvgTraceMedian(traceData[:,1], intervalSize, invert = invert)
		
		return (traceResult1, traceResult2)
	
		
	##########
	
	#Subtract offset from array.
	def pedestalSubtraction(self, traceData, pedestalOffset):

		returnData = traceData - pedestalOffset
		
		return returnData
	
	#Subtract offsets from two arrays.
	def pedestalDualSubtraction(self, traceData1, traceData2, pedestalOffset1, pedestalOffset2):
		# tracedata1/2 = np arrays, pedestalData = np array of pedestal sums
		returnData1 = self.pedestalSubtraction(traceData1, pedestalOffset1)
		returnData2 = self.pedestalSubtraction(traceData2, pedestalOffset2)
		
		return self.packageTrace(returnData1, returnData2)
		
	#Subtract from a list of numpy arrays.	
	def pedestalSubtractions(self, traceData, pedestaloffset):
		#traceData = list of np arrays, pedestalData = np array of pedestal sums
		returnData = []

		for i, element in enumerate(traceData):
			returnData.append(self.pedestalSubtraction(element,pedestaloffset))
			
		return returnData
	
	#Convenience for list of 2d numpy arrays
	def pedestalDualSubtractions(self, traceData, pedestaloffsets):
		# traceData = a list of np column stacks, pedestaloffsets = tuple of pedestalOffsets i.e. what is returned from
		#getDualAvgTraceMedian,
		#intervalSize = size of interval over which pedestal sums were made
		returnData = []
		
		for i,element in enumerate(traceData):
			returnData.append(self.pedestalDualSubtraction(element[:,0],element[:,1], pedestaloffsets[0], pedestaloffsets[1]))
			
		return returnData
	
		
	##########
	
	#Simple spike rejection technique. Subject to fine tuning by user. Rejects traces with X points over Y threshold.
	def spikeRejection(self, fullTraceData, testTraceData, voltageThreshold, timeThreshold, saveSpikes=False):
		
		returnData = []
		savedSpikes = []
		
		for i, element in enumerate(testTraceData):
			if len(element[:,0][np.where( element[:,0] < voltageThreshold)]) > timeThreshold:
				returnData.append(fullTraceData[i])
			elif saveSpikes == True:
				savedSpikes.append(fullTraceData[i])

		if saveSpikes == True:
			print(str(len(returnData))+' accepted, '+str(len(savedSpikes))+' rejected')
			return (returnData, savedSpikes)
		else:
			print(str(len(returnData))+' accepted')
			return returnData
		
	##########
	
	
	
		
class tracePlotter:
		
	def __init__(self, config = None):
		self.config = config if config is not None else sys.exit('No config found for traceExtractor. Aborting.')

	#Reads in values from config file and creates all forseeable variables.		
	def setupClassVariables(self):

		#Read in config variables
		self.workingDir = Path(self.config['General']['workingDir'])
		self.channel1 = self.workingDir/self.config['General']['channel1']
		self.channel2 = self.workingDir/self.config['General']['channel2']
		self.channel1BG = self.workingDir/self.config['General']['channel1BG']
		self.channel2BG = self.workingDir/self.config['General']['channel2BG']
		self.SIL = float(self.config['General']['signalIntegrationLower'])
		self.SIU = float(self.config['General']['signalIntegrationUpper'])
		self.PIL = float(self.config['General']['pedestalIntegrationLower'])
		self.PIU = float(self.config['General']['pedestalIntegrationUpper'])
		self.horizontalGridNumber = float(self.config['General']['horizontalGridNumber'])
		self.horizontalWidth = float(self.config['General']['horizontalWidth'])
		self.voltageThreshold = float(self.config['General']['voltageThreshold'])
		self.timeThreshold = int(self.config['General']['timeThreshold'])
		self.horizontalSampleNumber = float(self.config['General']['horizontalSampleNumber'])
		self.horizontalUnits = self.config['General']['horizontalUnits']
		self.verticalUnits = self.config['General']['verticalUnits']
		if self.horizontalUnits == 's':
			self.horizontalUnits = '$Seconds$'			
		elif self.horizontalUnits == 'm':
			self.horizontalUnits = '$milliseconds$'		
		elif self.horizontalUnits == 'u':
			self.horizontalUnits = '$\mu s$'
		elif self.horizontalUnits == 'n':
			self.horizontalUnits = '$ns$'
		self.verticalDivison = float(self.config['General']['verticalDivison'])
		self.verticalGridNumber  = float(self.config['General']['verticalGridNumber'])
		self.saveTextData = self.config['General']['saveTextData']
		self.savePrefix = self.config['General']['savePrefix']
		self.saveFolder = self.config['General']['saveFolder']
		
		#Calculate some values
		self.horizontalConversionPhys = self.horizontalWidth/self.horizontalGridNumber #Turns grid points/ticks into physicsal marks
		self.horizontalConversionImag = self.horizontalSampleNumber/self.horizontalWidth #Turns physical units into array index 
		self.horizontalSymmetricStart = self.horizontalWidth/2 - self.horizontalWidth
		self.horizontalSymmetricStop = self.horizontalWidth/2		
		self.horizontalStart = self.horizontalSymmetricStart*self.horizontalConversionPhys
		self.horizontalStop = self.horizontalSymmetricStop*self.horizontalConversionPhys
		self.iSIL = int((self.SIL+self.horizontalSymmetricStop)*self.horizontalConversionImag)
		self.iSIU = int((self.SIU+self.horizontalSymmetricStop)*self.horizontalConversionImag)
		self.iPIL = int((self.PIL+self.horizontalSymmetricStop)*self.horizontalConversionImag)
		self.iPIU = int((self.PIU+self.horizontalSymmetricStop)*self.horizontalConversionImag)
		self.signalLimits = [int(self.iSIL),int(self.iSIU)]
		self.pedestalLimits = [int(self.iPIL),int(self.iPIU)]
		self.signalInterval = self.iSIU - self.iSIL
		self.pedestalInterval = self.iPIU - self.iPIL
		self.veritcalDivisionStart = 2.
		self.verticalEnd = self.veritcalDivisionStart*self.verticalDivison
		self.veritcalStart = -1*(self.verticalGridNumber-self.veritcalDivisionStart)*self.verticalDivison
		
		#Print sanity check
		print('Time for a sanity check...!')
		print('Physical-to-image signal integration window map: [ '+str(self.SIL)+','+str(self.SIU)+' ] --> [ '+str(self.iSIL)+','+str(self.iSIU)+' ]')
		print('Physical-to-image pedestal integration window map: [ '+str(self.PIL)+','+str(self.PIU)+' ] --> [ '+str(self.iPIL)+','+str(self.iPIU)+' ]')

		

	def pedestalPlot(self, pedSum, PIL, PIU, legendObject, units):
		
		myFont = {'fontname':'Liberation Serif'}
		plt.figure(figsize=(9,6), dpi=100)
		bins = np.linspace(np.min(pedSum),np.max(pedSum),200)
		plt.title('Summed Pedestal Distribution (Interval: ['+str(PIL)+' '+units+', '+str(PIU)+' '+units+'])',**myFont)
		plt.ylabel('Number of Events [$N$]',**myFont)
		plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
		plt.hist(pedSum, bins, alpha=1., label='CsI Trace',color='blue')
		plt.legend(loc='upper right')
		plt.savefig('1_Summed_Pedestal_Distribution.png',dpi=500)
		plt.show()
		plt.close()

	def pedestalDualPlot(self, pedSum, PIL, PIU, legendObject1, legendObject2, units, fileName):
		
		myFont = {'fontname':'Liberation Serif'}
		plt.figure(figsize=(9,6), dpi=100)
		bins = np.linspace(np.min(np.array((np.min(pedSum[:,0]),np.min(pedSum[:,1])))),np.max(np.array((3*np.median(pedSum[:,0]),3*np.median(pedSum[:,1])))),200)
		plt.title('Summed Pedestal Distribution (Interval: ['+str(PIL)+' '+units+', '+str(PIU)+' '+units+'])',**myFont)
		plt.ylabel('Number of Events [$N$]',**myFont)
		plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
		plt.hist(pedSum[:,0], bins, alpha=1., label=legendObject1,color='blue')
		plt.hist(pedSum[:,1], bins, alpha=1., label=legendObject2,color='red')
		plt.legend(loc='upper right')
		plt.savefig(fileName,dpi=500)
		plt.show()
		plt.close()
		
	def plotPHD(self, sums, SIL, SIU, bins, title, legendObject, units, color, n, fileName):
		
		myFont = {'fontname':'Liberation Serif'}
		plt.figure(figsize=(9,6), dpi=100)
		plt.title(title+' (Interval: ['+str(SIL)+' '+units+', '+str(SIU)+' '+units+'])',**myFont)
		plt.ylabel('Number of Events [$N$]',**myFont)
		plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
		plt.hist(sums, bins, alpha=1.0, label=legendObject,color=color)
		plt.ylim(0,n)
		plt.legend(loc='upper right')
		plt.savefig(fileName,dpi=500)
		plt.show()
		plt.close()
		
	def plotDualPHD(self, sums1, sums2, SIL, SIU, bins, title, legendObject, units, color1, color2, n):
		
		myFont = {'fontname':'Liberation Serif'}
		plt.figure(figsize=(9,6), dpi=100)
		plt.title(title+' (Interval: ['+str(SIL)+' '+units+', '+str(SIU)+' '+units+'])',**myFont)
		plt.ylabel('Number of Events [$N$]',**myFont)
		plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
		plt.hist(sums1, bins, alpha=0.5, label=legendObject,color=color1)
		plt.hist(sums2, bins, alpha=0.5, label=legendObject,color=color2)
		plt.ylim(0,n)
		plt.legend(loc='upper right')
		plt.savefig('1_Less_Simple_Summed_PHD_Full.png',dpi=500)
		plt.show()
		plt.close()	
		
		
	def plotDualTrace(self, trace1, trace2, HStart, HStop, HSN, HGN, VS, VE, VGN,
		title, units, SIL, SIU, PIL, PIU, color1, color2, label1, label2, fileName):
		
		myFont = {'fontname':'Liberation Serif'}
		plt.figure(figsize=(9,6), dpi=100)
		plt.subplot(111)
		x = np.linspace(HStart,HStop,HSN)
		plt.plot(x, trace1, color=color1, linewidth=0.5, linestyle="-")	
		plt.plot(x, trace2, color=color2, linewidth=0.5, linestyle="-")
		plt.xticks(np.linspace(HStart, HStop, HGN+1,endpoint=True),**myFont)
		plt.ylim(VS,VE)
		plt.yticks(np.linspace(VS, VE, VGN+1, endpoint=True),**myFont)
		plt.grid(True)
		blue_patch = mpatches.Patch(color=color1, label=label1)
		red_patch = mpatches.Patch(color=color2, label=label2)
		mpl.rc('font',family='Liberation Serif')
		plt.legend(loc='lower right',handles=[red_patch,blue_patch])
		plt.title(title,**myFont)
		plt.xlabel('Time Relative to Trigger ['+units+']',**myFont)
		plt.ylabel('Voltage [$V$]',**myFont)		
		plt.plot([SIL,SIL],[VS,VE],color='yellow',linestyle="--",alpha=0.65)
		plt.plot([SIU,SIU],[VS,VE],color='yellow',linestyle="--",alpha=0.65)
		plt.plot([PIL,PIL],[VS,VE],color='purple',linestyle="--",alpha=0.65)
		plt.plot([PIU,PIU],[VS,VE],color='purple',linestyle="--",alpha=0.65)
		plt.savefig(fileName,dpi=500)
		plt.close()		
		
		