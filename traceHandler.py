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
import pandas as pd
from configparser import ConfigParser, ExtendedInterpolation
import os

def codePause():
	code.interact(local=locals())
	sys.exit('Code Break!')

#This class does the data extraction and data crunching. Mostly a collection of simple functions 
#so that main scripts can easily just call them and do analysis.
class traceExtractor:
	
	def __init__(self, config = None):
		self.config = config if config is not None else sys.exit('No config found for traceExtractor. Aborting.')

	#Reads in values from config and meta files and creates all forseeable variables.		
	def setupClassVariables(self):

		def _absoluteParsX(self):
			
			xRange = ( float(0), self.xWidthPhysical )
			xRangeImage = ( int(0),  int(self.sample))
			xTicks = np.linspace(xRange[0], xRange[1], self.xDivs+1)
			iSIL = int(round(self.SIL*self.xConversionPtoI))
			iSIU = int(round(self.SIU*self.xConversionPtoI))
			iPIL = int(round(self.PIL*self.xConversionPtoI))
			iPIU = int(round(self.PIU*self.xConversionPtoI))
			
			return (xRange, xRangeImage, xTicks, iSIL, iSIU, iPIL, iPIU)
			
			
		def _relativeParsX(self):
			
			xRange = ( -1*(self.xWidthPhysical/2-self.xLocation), self.xWidthPhysical/2+self.xLocation )
			xRangeImage = ( int(round(xRange[0]*self.xConversionPtoI)), int(round(xRange[1]*self.xConversionPtoI)))
			if self.xRelativeGrid == True:
				leftTick = -1*self.xWidthPhysical/self.xDivs
				rightTick =  self.xWidthPhysical/self.xDivs
				xTicks = np.array([0])
				while leftTick >= xRange[0]:
					xTicks = np.insert(xTicks, 0, leftTick)
					leftTick += -1*self.xWidthPhysical/self.xDivs
				while rightTick <= xRange[1]:
					xTicks = np.append(xTicks, rightTick)
					rightTick += self.xWidthPhysical/self.xDivs					
			elif self.xRelativeGrid == False:	
				xTicks = np.linspace(xRange[0], xRange[1], self.xDivs+1)
			#xRangeInterval = np.linspace(-1*self.xWidthPhysical/2, self.xWidthPhysical/2, self.xDivs+1)+self.xLocation
			iSIL = int(round((self.SIL+self.xLocation)*self.xConversionPtoI))
			iSIU = int(round((self.SIU+self.xLocation)*self.xConversionPtoI))
			iPIL = int(round((self.PIL+self.xLocation)*self.xConversionPtoI))
			iPIU = int(round((self.PIU+self.xLocation)*self.xConversionPtoI))
		
				
			return (xRange, xRangeImage, xTicks, iSIL, iSIU, iPIL, iPIU)
			
		def _symmetricParsX(self):
			
			xRange = ( -1*self.xWidthPhysical/2, self.xWidthPhysical/2 )
			xRangeImage = ( int(round(xRange[0]*self.xConversionPtoI)), int(round(xRange[1]*self.xConversionPtoI)))
			xTicks = np.linspace(xRange[0], xRange[1], self.xDivs+1)
			iSIL = int(round((self.SIL+xRange[1])*self.xConversionPtoI))
			iSIU = int(round((self.SIU+xRange[1])*self.xConversionPtoI))
			iPIL = int(round((self.PIL+xRange[1])*self.xConversionPtoI))
			iPIU = int(round((self.PIU+xRange[1])*self.xConversionPtoI))
		
			return (xRange, xRangeImage, xTicks, iSIL, iSIU, iPIL, iPIU)

		def _relativeParsY(self):
			yRange = (-1*self.scale*self.yDivs/2-self.yLocation1, self.scale*self.yDivs/2-self.yLocation1 )
			yTicks = np.linspace(yRange[0], yRange[1], self.xDivs+1)
			
			return (yRange, yTicks)
			
		def _symmetricParsY(self):
			yRange = ( -1*self.scale*self.yDivs/2, self.scale*self.yDivs/2 )
			yTicks = np.linspace(yRange[0], yRange[1], self.xDivs+1)
			
			return (yRange, yTicks)
			
		def _getPars(self, f):
			
			def wrapper():
				xRange, xRangeImage, xTicks, iSIL, iSIU, iPIL, iPIU = f(self)
				WindowPars = {
					'xRange' : xRange,
					'xRangeImage' : xRangeImage,
					'xTicks' : xTicks,
					'iSIL' : iSIL,
					'iSIU' : iSIU,
					'iPIL' : iPIL,
					'iPIU' : iPIU
					}
				return WindowPars
				
			return wrapper
		
			
		#Read in config variables
		#[General]
		self.workingDir = Path(self.config['General']['workingDir'])
		self.dataDir = Path(self.config['General']['dataDir'])	
		self.load = self.config['General'].getboolean('load')
		self.loadFrom = self.dataDir/self.config['General']['loadFrom']
		self.meta = self.dataDir/self.config['General']['meta']
		self.metaBG = self.dataDir/self.config['General']['metaBG']
		self.channel1 = self.dataDir/self.config['General']['channel1']
		self.channel2 = self.dataDir/self.config['General']['channel2']
		self.channel1BG = self.dataDir/self.config['General']['channel1BG']
		self.channel2BG = self.dataDir/self.config['General']['channel2BG']
		self.doubleChannel = self.config['General'].getboolean('doubleChannel')
		self.BGSubtraction = self.config['General'].getboolean('BGSubtraction')
		self.saveData = self.config['General'].getboolean('saveData')
		self.savePlots = self.config['General'].getboolean('savePlots')
		self.plotsFolder = self.config['General']['plotsFolder']
		self.saveID = self.config['General']['saveID']
		
		#[Window]
		self.xPlotType = self.config['Window']['xPlotType']
		self.xPlotSelection = {'absolute' : False, 'relative' : False, 'symmetric' : False}
		self.xPlotSelection[self.xPlotType.lower()] = True
		self.yPlotType = self.config['Window']['yPlotType']
		self.yPlotSelection = {'relative' : False, 'symmetric' : False}
		self.yPlotSelection[self.yPlotType.lower()] = True
		self.reproduceOscilliscope = self.config['Window'].getboolean('reproduceOscilliscope')
		self.xRelativeGrid = self.config['Window'].getboolean('xRelativeGrid')
		
		#meta file->[General]
		self.metaConfig = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
		self.metaConfig.read(str(self.meta))
		self.xWidthPhysical = float(self.metaConfig['General']['xWidthPhysical'])
		self.xWidthUnit = self.metaConfig['General']['xWidthUnit']
		self.yHeightUnits = self.metaConfig['General']['yHeightUnits']
		self.xLocation = float(self.metaConfig['General']['xLocation'])
		self.sample = float(self.metaConfig['General']['sample'])
		self.xDivs = float(self.metaConfig['General']['xDivs'])
		self.yDivs = float(self.metaConfig['General']['yDivs'])
		
		#[channel1]
		self.VoltsPerDiv1 = float(self.metaConfig['Channel1']['VoltsPerDiv'])
		self.yLocation1 = float(self.metaConfig['Channel1']['yLocation'])
		
		#[channel2]
		if self.doubleChannel == True:	
			self.VoltsPerDiv2 = float(self.metaConfig['Channel2']['VoltsPerDiv'])
			self.yLocation2 = float(self.metaConfig['Channel2']['yLocation'])
		#parse unit for plot axis
		if self.xWidthUnit == 's':
			self.xWidthUnit = '$Seconds$'			
		elif self.xWidthUnit == 'm':
			self.xWidthUnit = '$milliseconds$'		
		elif self.xWidthUnit == 'u':
			self.xWidthUnit = '$\mu s$'
		elif self.xWidthUnit == 'n':
			self.xWidthUnit = '$ns$'		
		
		#Repeat above for BG file
		if self.BGSubtraction == True:
			self.metaConfigBG = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
			self.metaConfigBG.read(str(self.metaBG))
			self.xWidthPhysicalBG = float(self.metaConfigBG['General']['xWidthPhysical'])
			self.xWidthUnitBG = self.metaConfigBG['General']['xWidthUnit']
			self.yHeightUnitsBG = self.metaConfigBG['General']['yHeightUnits']
			self.xLocationBG = float(self.metaConfigBG['General']['xLocation'])
			self.sampleBG = float(self.metaConfigBG['General']['sample'])
			self.xDivsBG = float(self.metaConfigBG['General']['xDivs'])
			self.yDivsBG = float(self.metaConfigBG['General']['yDivs'])
			self.VoltsPerDivBG1 = float(self.metaConfigBG['Channel1']['VoltsPerDiv'])
			self.yLocationBG1 = float(self.metaConfigBG['Channel1']['yLocation'])
			if self.doubleChannel == True:		
				self.VoltsPerDivBG2 = float(self.metaConfigBG['Channel2']['VoltsPerDiv'])
				self.yLocationBG2 = float(self.metaConfigBG['Channel2']['yLocation'])	

		#[Integration]
		self.SIL = float(self.config['Integration']['signalIntegrationLower'])
		self.SIU = float(self.config['Integration']['signalIntegrationUpper'])
		self.PIL = float(self.config['Integration']['pedestalIntegrationLower'])
		self.PIU = float(self.config['Integration']['pedestalIntegrationUpper'])
		
		#[SpikeRejection]
		self.voltageThreshold = float(self.config['SpikeRejection']['voltageThreshold'])
		self.timeThreshold = int(self.config['SpikeRejection']['timeThreshold'])

		#Calculate some values
		#Conversion factors
		self.xConversionDtoP = self.xWidthPhysical/self.xDivs #Turns division points/ticks into physicsal marks
		self.xConversionPtoI = self.sample/self.xWidthPhysical #Turns physical units into array index

		#Window specific parameters
		if self.xPlotSelection['absolute'] == True:
			self.windowParameters = _getPars(self, _absoluteParsX)()
		elif self.xPlotSelection['relative'] == True:
			self.windowParameters = _getPars(self, _relativeParsX)()
		elif self.xPlotSelection['symmetric'] == True:
			self.windowParameters = _getPars(self, _symmetricParsX)()
		else:
			sys.exit('Something wrong in getting the x-axis window parameters!')
			
			
		
		self.signalLimitsPhys = [self.SIL, self.SIU]
		self.signalLimitsImag = [self.windowParameters['iSIL'], self.windowParameters['iSIU']]
		self.pedestalLimitsPhys = [self.PIL, self.PIU]
		self.pedestalLimitsImag = [self.windowParameters['iPIL'], self.windowParameters['iPIU']]

		self.signalIntervalPhys = self.SIU - self.SIL
		self.signalIntervalImag = self.windowParameters['iSIU'] - self.windowParameters['iSIL']
		self.pedestalIntervalPhys = self.PIU - self.PIL
		self.pedestalIntervalImag = self.windowParameters['iPIU'] - self.windowParameters['iPIL']
	
		if self.doubleChannel == True:		
			self.scale = np.max([self.VoltsPerDiv1, self.VoltsPerDiv2])
			if self.BGSubtraction == True:
				self.scaleBG = np.max([self.VoltsPerDivBG1, self.VoltsPerDivBG2])
				self.scale = np.max([self.scale, self.scaleBG])
		else:
			self.scale = self.VoltsPerDiv1
			if self.BGSubtraction == True:
				self.scaleBG = self.VoltsPerDivBG1
				self.scale = np.max([self.VoltsPerDiv1, self.VoltsPerDivBG1])
				
		if self.yPlotSelection['relative'] == True:
			self.yRange, self.yTicks = _relativeParsY(self)
		elif self.yPlotSelection['symmetric'] == True:
			self.yRange, self.yTicks = _symmetricParsY(self)
		else:
			sys.exit('Something wrong in getting the y-axis window parameters!')
			
		
		#Print sanity check
		print('Time for a sanity check...!')
		print('Physical-to-image signal integration window map: [ '+str(self.SIL)+','+str(self.SIU)+' ] --> [ '+str(self.windowParameters['iSIL'])+','+str(self.windowParameters['iSIU'])+' ]')
		print('Physical-to-image pedestal integration window map: [ '+str(self.PIL)+','+str(self.PIU)+' ] --> [ '+str(self.windowParameters['iPIL'])+','+str(self.windowParameters['iPIU'])+' ]')

		print(self.windowParameters['xRange'])
		print(self.windowParameters['xTicks'])
		code.interact(local=locals())
		sys.exit('Code Break!')

		
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
	
	
	#Get subtrace based on limits
	def extractSubtraces(self, traceData, limit, invert=False):
		
		if invert==False:
			return traceData[:][limit[0]:limit[1]]
		else:
			return -1*traceData[:][limit[0]:limit[1]]
			
	def extractDualSubtraces(self, traceData, limits, invert=False):
		
		returnData = traceData.copy(deep=True)
		
		if invert==False:
			returnData[returnData.columns[0::2]] = returnData[returnData.columns[0::2]][limits[0][0]:limits[0][1]]
			returnData[returnData.columns[1::2]] = returnData[returnData.columns[1::2]][limits[1][0]:limits[1][1]]
			return returnData
		else:
			returnData[returnData.columns[0::2]] = -1*returnData[returnData.columns[0::2]][limits[0][0]:limits[0][1]]
			returnData[returnData.columns[1::2]] = -1*returnData[returnData.columns[1::2]][limits[1][0]:limits[1][1]]
			return returnData
		
	##########	

	#Sum the traces...yes this is simple.
	def sumTraces(self, traceData):
		
		return traceData[:].sum()
		

	##########	
	
	
	#Get the average of the median value
	def getAvgMedian(self, traceData, intervalSize, invert = False):
		
		if invert == False:
			return traceData.median()/intervalSize
		else:
			return -1*traceData.median()/intervalSize
			
	def getDualAvgMedian(self, traceData, intervalSizes, invert = False):
		
		if invert == False:
			return (traceData[0::2].median()/intervalSizes[0], traceData[1::2].median()/intervalSizes[1])
		else:
			return (-1*traceData[0::2].median()/intervalSizes[0], -1*traceData[1::2].median()/intervalSizes[1])
			
			
	##########
	
	#Correct for pedestal offsets		
	def pedestalDualSubtractions(self, traceData, pedestalOffsets):
		
		traceData[traceData.columns[0::2]] = traceData[traceData.columns[0::2]].subtract(pedestalOffsets[0], axis='columns')
		traceData[traceData.columns[1::2]] = traceData[traceData.columns[1::2]].subtract(pedestalOffsets[1], axis='columns')
		
		return traceData
		
		
	##########
	
	#Simple spike rejection
	def spikeRejection(self, traceData, limits, voltageThreshold, timeThreshold, saveSpikes=False):
		
		k=0
		j=0
		
		for i, columns in enumerate(traceData.columns[0::2]):
			if (len(np.where(traceData['channel1_'+str(i)][limits[0]:limits[1]] < voltageThreshold)[0]) <= timeThreshold) and saveSpikes==True:
				if k==0:
					savedSpikes= pd.DataFrame(traceData['channel1_'+str(i)])
					savedSpikes['channel2_'+str(i)] = traceData['channel2_'+str(i)]
				else:
					savedSpikes[['channel1_'+str(i),'channel2_'+str(i)]] = traceData[['channel1_'+str(i),'channel2_'+str(i)]]
				k += 1
			elif(len(np.where(traceData['channel1_'+str(i)][limits[0]:limits[1]] < voltageThreshold)[0]) > timeThreshold):
				if j==0:
					returnData = pd.DataFrame(traceData['channel1_'+str(i)])
					returnData['channel2_'+str(i)] = traceData['channel2_'+str(i)]
				else:
					returnData[['channel1_'+str(i),'channel2_'+str(i)]] = traceData[['channel1_'+str(i),'channel2_'+str(i)]]
				j += 1
		print(str(len(returnData.columns[0::2]))+' accepted, '+str(len(savedSpikes.columns[0::2]))+' rejected')
		
		if saveSpikes == True:
			return (returnData, savedSpikes)
		else:
			return returnData
			
		
	##########
	
	
	#Misc. functions
	def tracetoPandas(self, traceList, indexArray):
		
		label = 'channel1_'
		
		for i, element in enumerate(traceList):
			if i == 0:
				d = {label+str(i):element[:]}
				dataFrameReturned = pd.DataFrame(d, inde3x=indexArray)
			else:
				dataFrameReturned[label+str(i)] = element[:]
	
		return dataFrameReturned
	
	def dualTraceToPandas(self, traceList, indexArray):

		label1 = 'channel1_'
		label2 = 'channel2_'
		
		for i, element in enumerate(traceList):
			if i == 0:
				d = {label1+str(i):element[:,0], label2+str(i):element[:,1]}
				dataFrameReturned = pd.DataFrame(d,index=indexArray)				
			else:
				dataFrameReturned[label1+str(i)], dataFrameReturned[label2+str(i)] = [element[:,0],element[:,1]]
				
		return dataFrameReturned
		
		
	def initializeData(self):
		try:
			print('Moving into working directory: '+str(self.workingDir)+'...')
		except NameError:
			sys.exit('WorkingDir variable not found. Did you setupClassVariables?')
		
		os.chdir(str(self.workingDir))
		
		#Read in files, return (linesChannel1, linesChannel2)
		#set up window time interval
		#create the traceList
		if self.load == True:
			traceList = pd.read_hdf('traceList_'+self.saveID+'.h5')
		else:
			print('Reading in lines... (this takes a a lot of of memory!)')
			lines = self.openDualRawTraceFile(self.channel1, self.channel2)
			interval = np.linspace(self.x,self.horizontalSymmetricStopPhys,self.horizontalSampleNumber)
			traceList = self.dualTraceToPandas(self.extractDualRawTraces(lines), interval)
			del lines, interval

			#Save if needed.
			if self.saveData == True:
				traceList.to_hdf('traceList_'+self.saveID+'.h5', 'table')
				
		return traceList
		
class tracePlotter:
		
	def __init__(self, config = None):
		self.config = config if config is not None else sys.exit('No config found for traceExtractor. Aborting.')

	#Reads in values from config file and creates all forseeable variables.		
	def setupClassVariables(self):

		#Read in config variables
		#[General]
		self.workingDir = Path(self.config['General']['workingDir'])
		self.dataDir = Path(self.config['General']['dataDir'])
		self.channel1 = self.dataDir/self.config['General']['channel1']
		self.channel2 = self.dataDir/self.config['General']['channel2']
		self.channel1BG = self.dataDir/self.config['General']['channel1BG']
		self.channel2BG = self.dataDir/self.config['General']['channel2BG']
		self.saveData = self.config['General'].getboolean('saveData')
		self.savePlots = elf.config['General'].getboolean('savePlots')
		self.dataFolder = self.config['General']['dataFolder']
		self.plotsFolder = self.config['General']['plotsFolder']
		self.saveFilename = self.config['General']['saveFilename']
		
		#[Window]
		self.symmetric = self.config['Window'].getboolean('symmetric')
		self.horizontalWidth = float(self.config['Window']['horizontalWidth'])
		self.horizontalGridNumber = float(self.config['Window']['horizontalGridNumber'])
		self.horizontalSampleNumber = float(self.config['Window']['horizontalSampleNumber'])
		self.horizontalUnits = self.config['Window']['horizontalUnits']
		if self.horizontalUnits == 's':
			self.horizontalUnits = '$Seconds$'			
		elif self.horizontalUnits == 'm':
			self.horizontalUnits = '$milliseconds$'		
		elif self.horizontalUnits == 'u':
			self.horizontalUnits = '$\mu s$'
		elif self.horizontalUnits == 'n':
			self.horizontalUnits = '$ns$'		
		self.verticalUnits = self.config['Window']['verticalUnits']
		self.verticalDivison = float(self.config['Window']['verticalDivison'])
		self.verticalGridNumber  = float(self.config['Window']['verticalGridNumber'])

		#[Integration]
		self.SIL = float(self.config['Integration']['signalIntegrationLower'])
		self.SIU = float(self.config['Integration']['signalIntegrationUpper'])
		self.PIL = float(self.config['Integration']['pedestalIntegrationLower'])
		self.PIU = float(self.config['Integration']['pedestalIntegrationUpper'])
		
		#[SpikeRejection]
		self.voltageThreshold = float(self.config['SpikeRejection']['voltageThreshold'])
		self.timeThreshold = int(self.config['SpikeRejection']['timeThreshold'])

		#Calculate some values
		#Conversion factors and array starts and stops
		self.horizontalConversionGtoP = self.horizontalWidth/self.horizontalGridNumber #Turns grid points/ticks into physicsal marks
		self.horizontalConversionPtoI = self.horizontalSampleNumber/self.horizontalWidth #Turns physical units into array index 
		self.horizontalSymmetricStartPhys = self.horizontalWidth/2 - self.horizontalWidth
		self.horizontalSymmetricStopPhys = self.horizontalWidth/2
		self.horizontalStartPhys = float(0)
		self.horizontalStopPhys = self.horizontalWidth
		self.horizontalStartImag = int(0)
		self.horizontalStopImag = int(self.horizontalSampleNumber)

		#Image integration limits (user gives physical location)
		if self.symmetric == True:
			self.iSIL = int(round((self.SIL+self.horizontalSymmetricStopPhys)*self.horizontalConversionPtoI))
			self.iSIU = int(round((self.SIU+self.horizontalSymmetricStopPhys)*self.horizontalConversionPtoI))
			self.iPIL = int(round((self.PIL+self.horizontalSymmetricStopPhys)*self.horizontalConversionPtoI))
			self.iPIU = int(round((self.PIU+self.horizontalSymmetricStopPhys)*self.horizontalConversionPtoI))
		else:
			self.iSIL = int(round(self.SIL*self.horizontalConversionPtoI))
			self.iSIU = int(round(self.SIU*self.horizontalConversionPtoI))
			self.iPIL = int(round(self.PIL*self.horizontalConversionPtoI))
			self.iPIU = int(round(self.PIU*self.horizontalConversionPtoI))
		
		self.signalLimitsPhys = [self.SIL, self.SIU]
		self.signalLimitsImag = [self.iSIL, self.iSIU]
		self.pedestalLimitsPhys = [self.PIL, self.PIU]
		self.pedestalLimitsImag = [self.iPIL, self.iPIU]

		self.signalIntervalPhys = self.SIU - self.SIL
		self.signalIntervalImag = self.iSIU - self.iSIL
		self.pedestalIntervalPhys = self.PIU - self.PIL
		self.pedestalIntervalImag = self.iPIU - self.iPIL
		
		self.veritcalDivisionStart = 2.
		self.verticalEnd = self.veritcalDivisionStart*self.verticalDivison
		self.veritcalStart = -1*(self.verticalGridNumber-self.veritcalDivisionStart)*self.verticalDivison
		
		#Print sanity check
		print('Time for a sanity check...!')
		print('Physical-to-image signal integration window map: [ '+str(self.SIL)+','+str(self.SIU)+' ] --> [ '+str(self.iSIL)+','+str(self.iSIU)+' ]')
		print('Physical-to-image pedestal integration window map: [ '+str(self.PIL)+','+str(self.PIU)+' ] --> [ '+str(self.iPIL)+','+str(self.iPIU)+' ]')

		
		
	def pedestalDualPlot(self, pedSum, PIL, PIU, legendObject1, legendObject2, units, fileName):
		
		myFont = {'fontname':'Liberation Serif'}
		plt.figure(figsize=(9,6), dpi=100)
		bins = np.linspace(pedSum.min(),np.max(3*pedSum[0::2].median(),3*pedSum[1::2].median()),200)
		plt.title('Summed Pedestal Distribution (Interval: ['+str(PIL)+' '+units+', '+str(PIU)+' '+units+'])',**myFont)
		plt.ylabel('Number of Events [$N$]',**myFont)
		plt.xlabel('-1$\cdot$Summed Voltage[$mV$]',**myFont)
		plt.hist(pedSum[0::2], bins, alpha=1., label=legendObject1,color='blue')
		plt.hist(pedSum[1::2], bins, alpha=1., label=legendObject2,color='red')
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
		
		
	def plotDualTrace(self, trace1, trace2, HStart, HStop, HSN, HGN, VS, VE, VGN,
		title, units, SIL, SIU, PIL, PIU, color1, color2, label1, label2, fileName):
		
		myFont = {'fontname':'Liberation Serif'}
		plt.figure(figsize=(9,6), dpi=100)
		plt.subplot(111)
		#code.interact(local=locals())
		#sys.exit('Code Break!')
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
