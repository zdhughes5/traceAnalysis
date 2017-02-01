"""
Created on Thu Jan 26 17:45:11 2017

@author: zdhughes
"""

import sys
import numpy as np
from pathlib import Path
import code

class traceExtractor:
	
	def __init__(self, config = None):
		self.config = config if config is not None else sys.exit('No config found for traceExtractor. Aborting.')
		self.workingDir = Path(self.config['General']['workingDir'])
		self.channel1 = self.workingDir/self.config['General']['channel1']
		self.channel2 = self.workingDir/self.config['General']['channel2']
		self.SIL = float(config['General']['signalIntegrationLower'])
		self.SIU = float(config['General']['signalIntegrationUpper'])
		self.PIL = float(config['General']['pedestalIntegrationLower'])
		self.PIU = float(config['General']['pedestalIntegrationUpper'])
		self.horizontalGridNumber = float(config['General']['horizontalGridNumber'])
		self.horizontalWidth = float(config['General']['horizontalWidth'])
		self.voltageThreshold = float(config['General']['voltageThreshold'])
		self.timeThreshold = int(config['General']['timeThreshold'])
		self.horizontalSampleNumber = float(config['General']['horizontalSampleNumber'])
		self.horizontalConversionPhys = self.horizontalWidth/self.horizontalGridNumber #Turns grid points/ticks into physicsal marks
		self.horizontalConversionImag = self.horizontalSampleNumber/self.horizontalWidth #Turns physical units into array index 
		self.horizontalSymmetricStart = self.horizontalGridNumber/2 - self.horizontalGridNumber
		self.horizontalSymmetricStop = self.horizontalGridNumber/2		
		self.horizontalStart = self.horizontalSymmetricStart*self.horizontalConversionPhys
		self.horizontalStop = self.horizontalSymmetricStop*self.horizontalConversionPhys
		self.iSIL = (self.SIL+self.horizontalSymmetricStop)*self.horizontalConversionImag
		self.iSIU = (self.SIU+self.horizontalSymmetricStop)*self.horizontalConversionImag
		self.iPIL = (self.PIL+self.horizontalSymmetricStop)*self.horizontalConversionImag
		self.iPIU = (self.PIU+self.horizontalSymmetricStop)*self.horizontalConversionImag
		self.signalLimits = [int(self.iSIL),int(self.iSIU)]
		self.pedestalLimits = [int(self.iPIL),int(self.iPIU)]
		self.signalInterval = self.iSIU - self.iSIL
		self.pedestalInterval = self.iPIU - self.iPIL
		
	def packageTrace(self, trace1, trace2):
		
		return np.column_stack((trace1,trace2))
		
	def openRawTraceFile(self, channel1File):
		with open(str(channel1File),'r') as f:
			lines = [line.strip('\n') for line in f]
													
		return lines
		
	def openDualRawTraceFile(self, channel1File, channel2File):
		
		lines1 = self.openRawTraceFile(channel1File)
		lines2 = self.openRawTraceFile(channel2File)
													
		return (lines1, lines2)		
	
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
		
	def extractDualRawTraces(self, lines, saveTraces=False, traceFilename = 'trace_'):
		
		channel1Data = self.extractRawTraces(lines[0])
		channel2Data = self.extractRawTraces(lines[1])
		returnData = []
		
		for i, element in enumerate(channel1Data):
			returnData.append(self.packageTrace(channel1Data[i],channel2Data[i]))
			if saveTraces == True:
				np.savetext(traceFilename+str(i)+returnData[i]+'.txt',fmt='%.6e')
		
		return returnData
			
	def extractSubtrace(self, traceData, limits, invert= False):
		
		#Takes in a np array/list and pulls out subarray based on limits
		if invert == True:
			returnData = -1*traceData[limits[0]:limits[1]]
		else:
			returnData = traceData[limits[0]:limits[1]]
			
		return returnData
		
	def extractDualSubtrace(self, traceData, limits, invert = False):
		
		#Takes in two single np arrays/lists and pulls out subarrays based on limits
		returnData1 = self.extractSubtrace(traceData[:,0],limits, invert = invert)
		returnData2 = self.extractSubtrace(traceData[:,1],limits, invert = invert)
		
		return self.packageTrace(returnData1,returnData2)
		
	def extractSubtraces(self, traceData, limits, invert = False):
		
		#Takes in a list of np arrays/lists and pulls out subarrays based on limits
		returnData = []
		
		for i, element in enumerate(traceData):
			if invert == False:
				returnData.append(self.extractSubtrace(element, limits))
			elif invert == True:
				returnData.append(-1*self.extractSubtrace(element, limits))
		
		return returnData
		
	def extractDualSubtraces(self, traceData, limits, invert = False):
		
		#Takes in a list of np column stacks and pulls out subarrays based on limits		
		returnData = []
		#Iterate over list length
		for i, element in enumerate(traceData):
			returnData.append(self.extractDualSubtrace(element, limits, invert = invert))
			
		return returnData
		
	def sumTrace(self, traceData):
		
		return np.sum(traceData)
		
	def sumTraces(self, traceData):
		
		returnData = []
		
		for i, element in enumerate(traceData):
			returnData.append(np.sum(element))
			
		return returnData
		
	def sumDualTrace(self, traceData1, traceData2):
		
		returnData1 = self.sumTrace(traceData1)
		returnData2 = self.sumTrace(traceData2)
		
		return self.packageTrace(returnData1,returnData2)
		
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
	
	def getTraceMedian(self, traceData):
		#median on np array, like sums
		return np.median(traceData)
				
	def getAvgTraceMedian(self, traceData, intervalSize, invert = False):
		#traceData = np array of summed values

		if invert  == True:
			traceMedian = self.getTraceMedian(traceData)
			return -1*(traceMedian/intervalSize)
		elif invert  == False:
			traceMedian = self.getTraceMedian(traceData)
			return traceMedian/intervalSize
		
	def getDualAvgTraceMedian(self, traceData, intervalSize, invert = False):
		
		#traceData = np column stack of summed values
		traceResult1 = self.getAvgTraceMedian(traceData[:,0], intervalSize, invert = invert)
		traceResult2 = self.getAvgTraceMedian(traceData[:,1], intervalSize, invert = invert)
		
		return (traceResult1, traceResult2)
		
	def pedestalSubtraction(self, traceData, pedestalOffset):

		returnData = traceData - pedestalOffset
		
		return returnData
		
	def pedestalDualSubtraction(self, traceData1, traceData2, pedestalOffset1, pedestalOffset2):
		# tracedata1/2 = np arrays, pedestalData = np array of pedestal sums
		returnData1 = self.pedestalSubtraction(traceData1, pedestalOffset1)
		returnData2 = self.pedestalSubtraction(traceData2, pedestalOffset2)
		
		return self.packageTrace(returnData1, returnData2)
		
		
	def pedestalSubtractions(self, traceData, pedestaloffset):
		#traceData = list of np arrays, pedestalData = np array of pedestal sums
		returnData = []

		for i, element in enumerate(traceData):
			returnData.append(self.pedestalSubtraction(element,pedestaloffset))
			
		return returnData
		
	def pedestalDualSubtractions(self, traceData, pedestaloffsets):
		# traceData = a list of np column stacks, pedestaloffsets = tuple of pedestalOffsets i.e. what is returned from
		#getDualAvgTraceMedian,
		#intervalSize = size of interval over which pedestal sums were made
		returnData = []
		
		for i,element in enumerate(traceData):
			returnData.append(self.pedestalDualSubtraction(element[:,0],element[:,1], pedestaloffsets[0], pedestaloffsets[1]))
			
		return returnData
		
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
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	
	