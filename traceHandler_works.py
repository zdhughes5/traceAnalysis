"""
Created on Thu Jan 26 17:45:11 2017

@author: zdhughes
"""

import sys
import numpy as np

class traceExtractor:
	
	def __init__(self, config = None):
		self.config = config if config is not None else sys.exit('No config found for traceExtractor. Aborting.')		
		
		
	def openRawTraceFile(self, channel1File):
		with open(str(channel1File),'r') as f:
			lines = [line.strip('\n') for line in f]
													
		return lines
	
	def extractRawTrace(self, lines, saveTraces=False, traceFilename = 'trace_'):
		
		totalTraces = len([x for x in lines if x == 'data:'])
		traceList = []

		for i, trash  in enumerate(range(totalTraces)):

			lines[7+i*10007:10007+10007]
			extractedTrace = np.array(list(map(float,lines[7+i*10007:10007+i*10007])))
			traceList.append(extractedTrace)
			
			if saveTraces == True:
				np.savetxt(traceFilename+str(i)+'.txt',traceList[i],fmt='%.6e')
			
		return traceList
		
	def packageTrace(self, trace1, trace2):
		
		return np.column_stack((trace1,trace2))
		
	def packageTraces(self, traces1, traces2, saveTraces=False, traceFilename = 'trace_'):
		
		packagedTrace = []
		for i, trash in enumerate(traces1):
			
			packagedTrace.append(np.column_stack((traces1[i],traces2[i])))

			if saveTraces == True:
				np.savetxt(traceFilename+str(i)+'.txt',packagedTrace[i],fmt='%.6e')			

		return packagedTrace
		
		
	def extractDualRawTraces(self, channel1, channel2, saveTraces=False, traceFilename = 'trace_'):
		
		lines1 = self.openRawTraceFile(channel1)
		lines2 = self.openRawTraceFile(channel2)
		
		channel1Data = self.extractRawTrace(lines1)
		channel2Data = self.extractRawTrace(lines2)
		
		tracePackage = self.packageTraces(channel1Data, channel2Data, saveTraces=saveTraces, traceFilename=traceFilename)
		
		return tracePackage
		
	def extractSubtrace(self, traceData, limits, invert = False):
		
		#Takes in a np array/list and pulls out subarray based on limits
		if invert == True:
			returnData = -1*traceData[limits[0]:limits[1]]
		else:
			returnData = traceData[limits[0]:limits[1]]
			
		return returnData
		
	def extractDualSubtrace(self, traceData1, traceData2, limits, invert = False):
		
		#Takes in two single np arrays/lists and pulls out subarrays based on limits
		returnData1 = self.extractSubtrace(traceData1,limits, invert = invert)
		returnData2 = self.extractSubtrace(traceData2,limits, invert = invert)
		
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
			returnData.append(self.extractDualSubtrace(element[:,0],element[:,1],limits, invert = invert))
			
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
				
	def getAvgTraceMedian(self, traceData, intervalSize):
		#traceData = np array of summed values
		
		traceMedian = self.getTraceMedian(traceData)
		return traceMedian/intervalSize
				
	def getDualAvgTraceMedian(self, traceData, intervalSize):
		
		#traceData = np column stack of summed values
		traceResult1 = self.getAvgTraceMedian(traceData[:,0], intervalSize)
		traceResult2 = self.getAvgTraceMedian(traceData[:,1], intervalSize)
		
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
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	
	