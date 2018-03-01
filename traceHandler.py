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
import matplotlib.lines as mlines
import pandas as pd
from configparser import ConfigParser, ExtendedInterpolation
import subprocess
from math import sqrt
from scipy import signal
from scipy.optimize import curve_fit

#This class does the data extraction, crunching, and plotting. Data crunching is done functionally. 
#That is, even though the class functions are not static, they are called like they are.
#Why? I'm self-taught and experimenting.
class traceExtractor:
	
	def __init__(self, config = None, c = None):
		self.config = config if config is not None else sys.exit('No config found for traceExtractor. Aborting.')
		self.c = c if c is not None else colors()
		
	#Reads in values from config and meta files and creates all forseeable *static* variables.
	#meta file = file that describes oscilliscope settings	
	def setupClassVariables(self):
		
		def getDefaultColor(self):
			self.currentColor += 1
			if self.currentColor == len(self.defaultColors):
				self.currentColor = -1
			return self.defaultColors[self.currentColor]

		def _absoluteParsX(self):
			
			xRange = ( float(0), self.xWidthPhysical )
			xRangeImage = ( int(0),  int(self.sample))
			xTicks = np.linspace(xRange[0], xRange[1], self.xDivs+1)
			iSIL = int(round(self.SIL*self.xConversionPtoI))
			iSIU = int(round(self.SIU*self.xConversionPtoI))
			iPIL = int(round(self.PIL*self.xConversionPtoI))
			iPIU = int(round(self.PIU*self.xConversionPtoI))
			x = np.linspace(xRange[0], xRange[1], self.sample)
			xWidthUnit = self.xWidthUnit
			selection = 'absolute'
		
			return (xRange, xRangeImage, x, xTicks, iSIL, iSIU, iPIL, iPIU, xWidthUnit, selection)
			
			
		def _relativeParsX(self):
			
			xRange = ( -1*(self.xWidthPhysical/2+self.xLocation), self.xWidthPhysical/2-self.xLocation )
			xRangeImage = ( int(0),  int(self.sample))
			if self.xRelativeGrid == True:
				leftTick = -1*self.xWidthPhysical/self.xDivs
				rightTick =  self.xWidthPhysical/self.xDivs
				xTicks = np.array([0])
				while rightTick <= xRange[1]:
					xTicks = np.append(xTicks, rightTick)
					rightTick += self.xWidthPhysical/self.xDivs	
				while leftTick >= xRange[0]:
					xTicks = np.insert(xTicks, 0, leftTick)
					leftTick += -1*self.xWidthPhysical/self.xDivs				
			elif self.xRelativeGrid == False:	
				xTicks = np.linspace(xRange[0], xRange[1], self.xDivs+1)			
			iSIL = int(round((self.SIL+self.xWidthPhysical/2+self.xLocation)*self.xConversionPtoI))
			iSIU = int(round((self.SIU+self.xWidthPhysical/2+self.xLocation)*self.xConversionPtoI))
			iPIL = int(round((self.PIL+self.xWidthPhysical/2+self.xLocation)*self.xConversionPtoI))
			iPIU = int(round((self.PIU+self.xWidthPhysical/2+self.xLocation)*self.xConversionPtoI))
			x = np.linspace(xRange[0], xRange[1], self.sample)
			xWidthUnit = self.xWidthUnit
			selection = 'relative'
		
			return (xRange, xRangeImage, x, xTicks, iSIL, iSIU, iPIL, iPIU, xWidthUnit, selection)
			
		def _symmetricParsX(self):
			
			xRange = ( -1*self.xWidthPhysical/2, self.xWidthPhysical/2 )
			xRangeImage = ( int(0),  int(self.sample))
			xTicks = np.linspace(xRange[0], xRange[1], self.xDivs+1)
			iSIL = int(round((self.SIL+self.xWidthPhysical/2)*self.xConversionPtoI))
			iSIU = int(round((self.SIU+self.xWidthPhysical/2)*self.xConversionPtoI))
			iPIL = int(round((self.PIL+self.xWidthPhysical/2)*self.xConversionPtoI))
			iPIU = int(round((self.PIU+self.xWidthPhysical/2)*self.xConversionPtoI))
			x = np.linspace(xRange[0], xRange[1], self.sample)
			xWidthUnit = self.xWidthUnit
			selection = 'symmetric'
		
			return (xRange, xRangeImage, x, xTicks, iSIL, iSIU, iPIL, iPIU, xWidthUnit, selection)

		def _relativeParsY(self, scale):
			yRange = (-1*scale*self.yDivs/2-self.yLocation1*scale, scale*self.yDivs/2-self.yLocation1*scale )
			yTicks = np.linspace(yRange[0], yRange[1], self.yDivs+1)
			
			return (yRange, yTicks)
			
		def _symmetricParsY(self, scale):
			yRange = ( -1*scale*self.yDivs/2, scale*self.yDivs/2 )
			yTicks = np.linspace(yRange[0], yRange[1], self.yDivs+1)
			
			return (yRange, yTicks)
			
		def _getParsX(self, f):
			
			def wrapper():
				
				xRange, xRangeImage, x, xTicks, iSIL, iSIU, iPIL, iPIU, xWidthUnit, selection = f(self)
				
				signalLimitsPhys = (self.SIL, self.SIU)
				signalLimitsImag = (iSIL, iSIU)
				pedestalLimitsPhys = (self.PIL, self.PIU)
				pedestalLimitsImag = (iPIL, iPIU)

				signalIntervalPhys = self.SIU - self.SIL
				signalIntervalImag = iSIU - iSIL
				pedestalIntervalPhys = self.PIU - self.PIL
				pedestalIntervalImag = iPIU - iPIL
				
				windowPars = {
					'xRange' : xRange,
					'xTicks' : xTicks,
					'x' : x,
					'SIL' : self.SIL,
					'SIU' : self.SIU,
					'PIL' : self.PIL,
					'PIU' : self.PIU,
					'signalLimits' : signalLimitsPhys,
					'pedestalLimits' : pedestalLimitsPhys,
					'signalInterval' : signalIntervalPhys,
					'pedestalInterval' : pedestalIntervalPhys,
					'xWidthUnit' : xWidthUnit,
					'selection' : selection
					}
				dataPars = {
					'xRange' : xRangeImage,
					'SIL' : iSIL,
					'SIU' : iSIU,
					'PIL' : iPIL,
					'PIU' : iPIU,
					'signalLimits' : signalLimitsImag,
					'pedestalLimits' : pedestalLimitsImag,
					'signalInterval' : signalIntervalImag,
					'pedestalInterval' : pedestalIntervalImag
					}	
					
				return windowPars, dataPars
				
			return wrapper
			
		def _getParsY(self, f, scale, objectName):
			
			def wrapper():
				
				yRange, yTicks = f(self, scale)
				windowPars = {
					'yRange' : yRange,
					'yTicks' : yTicks,
					'scale' : scale,
					'object' : objectName,
					'color' : getDefaultColor(self)
					}
				dataPars = {
					}
					
				return windowPars, dataPars
				
			return wrapper
		

		##############################			
		#Set-up variables
		##############################
			
		#Parse master file
		#[General]
		self.workingDir = Path(self.config['General']['workingDir'])
		self.dataDir = Path(self.config['General']['dataDir'])
		self.plotsDir = Path(self.config['General']['plotsDir'])
		self.traceDir = Path(self.config['General']['traceDir'])
		self.ansiColors = self.config['General'].getboolean('ansiColors')
		self.doPlots = self.config['General'].getboolean('doPlots')
		
		#[IO]
		self.saveData = self.config['IO'].getboolean('saveData')
		self.saveTo = Path(self.config['IO']['saveTo'])
		self.load = self.config['IO'].getboolean('load')
		self.loadFrom = Path(self.config['IO']['loadFrom'])
		self.showPlots = self.config['IO'].getboolean('showPlots')
		self.savePlots = self.config['IO'].getboolean('savePlots')
		self.allPlots = self.config['IO'].getboolean('allPlots')	
		
		#[Channels]
		self.doubleChannel = self.config['Channels'].getboolean('doubleChannel')
		self.BGSubtraction = self.config['Channels'].getboolean('BGSubtraction')
		self.channel1 = Path(self.config['Channels']['channel1'])
		self.channel2 = Path(self.config['Channels']['channel2'])
		self.meta = Path(self.config['Channels']['meta'])
		self.channel1BG = Path(self.config['Channels']['channel1BG'])
		self.channel2BG = Path(self.config['Channels']['channel2BG'])
		self.metaBG = Path(self.config['Channels']['metaBG'])
		
		#[Window]
		self.xPlotType = self.config['Window']['xPlotType']
		self.yPlotType = self.config['Window']['yPlotType']
		self.xRelativeGrid = self.config['Window'].getboolean('xRelativeGrid')

		#[Integration]
		self.SIL = float(self.config['Integration']['signalIntegrationLower'])
		self.SIU = float(self.config['Integration']['signalIntegrationUpper'])
		self.PIL = float(self.config['Integration']['pedestalIntegrationLower'])
		self.PIU = float(self.config['Integration']['pedestalIntegrationUpper'])
		
		#[SpikeRejection]
		self.doSpikeRejection = self.config['SpikeRejection'].getboolean('doSpikeRejection')
		self.voltageThreshold = float(self.config['SpikeRejection']['voltageThreshold'])
		self.timeThreshold = int(self.config['SpikeRejection']['timeThreshold'])
		
		#[SmoothedDoubleRejection]
		self.doDoubleRejection = self.config['SmoothedDoubleRejection'].getboolean('doDoubleRejection')
		self.SGWindow = int(self.config['SmoothedDoubleRejection']['SGWindow'])
		self.SGOrder = int(self.config['SmoothedDoubleRejection']['SGOrder'])
		self.minimaWindowDR = int(self.config['SmoothedDoubleRejection']['minimaWindowDR'])
		self.medianFactorDR = float(self.config['SmoothedDoubleRejection']['medianFactorDR'])
		self.fitWindow = int(self.config['SmoothedDoubleRejection']['fitWindow'])
		self.alphaThreshold = float(self.config['SmoothedDoubleRejection']['alphaThreshold'])
		
		#[PeakFinder]
		self.photonFilename = self.config['PeakFinder']['photonFilename']
		self.doPeakFinder = self.config['PeakFinder'].getboolean('doPeakFinder')
		self.savePhotons = self.config['PeakFinder'].getboolean('savePhotons')
		self.medianFactorPF = float(self.config['PeakFinder']['medianFactorPF'])
		self.stdFactor = float(self.config['PeakFinder']['stdFactor'])
		self.convWindow = int(self.config['PeakFinder']['convWindow'])
		self.convPower = float(self.config['PeakFinder']['convPower'])
		self.convSig = float(self.config['PeakFinder']['convSig'])
		self.minimaWindowPF = int(self.config['PeakFinder']['minimaWindowPF'])
		
		#[PhotonCounting]
		self.doPhotonCounting = self.config['PhotonCounting'].getboolean('doPhotonCounting')
		self.photonFiles = self.config['PhotonCounting']['photonFiles'].split(',')
		
		#Parse meta file		
		#[General]
		self.metaConfig = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
		self.metaConfig.read(str(self.meta))
		self.xWidthPhysical = float(self.metaConfig['General']['xWidthPhysical'])
		self.xWidthUnit = self.metaConfig['General']['xWidthUnit']
		self.yHeightUnits = self.metaConfig['General']['yHeightUnits']
		self.xLocation = -1*float(self.metaConfig['General']['xLocation'])
		self.sample = float(self.metaConfig['General']['sample'])
		self.xDivs = float(self.metaConfig['General']['xDivs'])
		self.yDivs = float(self.metaConfig['General']['yDivs'])
		
		#[channel1]
		self.object1 = self.metaConfig['Channel1']['object']
		self.VoltsPerDiv1 = float(self.metaConfig['Channel1']['VoltsPerDiv'])
		self.yLocation1 = float(self.metaConfig['Channel1']['yLocation'])
		
		#[channel2]
		if self.doubleChannel == True:
			self.object2 = self.metaConfig['Channel2']['object']
			self.VoltsPerDiv2 = float(self.metaConfig['Channel2']['VoltsPerDiv'])
			self.yLocation2 = float(self.metaConfig['Channel2']['yLocation'])
		
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
			
			self.objectBG1 = self.metaConfigBG['Channel1']['object']
			self.VoltsPerDivBG1 = float(self.metaConfigBG['Channel1']['VoltsPerDiv'])
			self.yLocationBG1 = float(self.metaConfigBG['Channel1']['yLocation'])
			
			if self.doubleChannel == True:
				self.objectBG2 = self.metaConfigBG['Channel2']['object']
				self.VoltsPerDivBG2 = float(self.metaConfigBG['Channel2']['VoltsPerDiv'])
				self.yLocationBG2 = float(self.metaConfigBG['Channel2']['yLocation'])	
		
		##############################
		#Calculate some values
		##############################
		
		#Get Axis styles
		self.xPlotSelection = {'absolute' : False, 'relative' : False, 'symmetric' : False}
		self.xPlotSelection[self.xPlotType.lower()] = True
		self.yPlotSelection = {'relative' : False, 'symmetric' : False}
		self.yPlotSelection[self.yPlotType.lower()] = True
		self.defaultColors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']
		self.currentColor = -1 
		self.photonColors = [getDefaultColor(self) for x in self.photonFiles]
		self.currentColor = -1

		#parse unit for plot axis
		if self.xWidthUnit == 's':
			self.xWidthUnit = '$Seconds$'			
		elif self.xWidthUnit == 'm':
			self.xWidthUnit = '$milliseconds$'		
		elif self.xWidthUnit == 'u':
			self.xWidthUnit = '$\mu s$'
		elif self.xWidthUnit == 'n':
			self.xWidthUnit = '$ns$'			
		
		#Conversion factors
		self.xConversionDtoP = self.xWidthPhysical/self.xDivs #Turns division points/ticks into physicsal marks
		self.xConversionPtoI = self.sample/self.xWidthPhysical #Turns physical units into array index

		#Window, data specific parameters
		if self.xPlotSelection['absolute'] == True:
			self.windowParametersX, self.dataParametersX = _getParsX(self, _absoluteParsX)()
		elif self.xPlotSelection['relative'] == True:
			self.windowParametersX, self.dataParametersX = _getParsX(self, _relativeParsX)()
		elif self.xPlotSelection['symmetric'] == True:
			self.windowParametersX, self.dataParametersX = _getParsX(self, _symmetricParsX)()
		else:
			sys.exit('Something wrong in getting the x-axis window parameters!')
				
		if self.yPlotSelection['relative'] == True:
			self.windowParametersY1, self.dataParametersY1 = _getParsY(self, _relativeParsY, self.VoltsPerDiv1, self.object1)()
			self.windowParametersY2, self.dataParametersY2 = _getParsY(self, _relativeParsY, self.VoltsPerDiv2, self.object2)()
		elif self.yPlotSelection['symmetric'] == True:
			self.windowParametersY1, self.dataParametersY1 = _getParsY(self, _symmetricParsY, self.VoltsPerDiv1)()
			self.windowParametersY2, self.dataParametersY2 = _getParsY(self, _symmetricParsY, self.VoltsPerDiv2)()
		else:
			sys.exit('Something wrong in getting the y-axis window parameters!')	
		
		##############################			
		#Print sanity check
		##############################
		print('\n#################################################################')
		print('################## '+self.c.lightgreen('Time for a sanity check...!')+' ##################')
		print('#################################################################\n')
		print('Physical-to-image signal integration window map: '+self.c.cyan('[ '+str(self.windowParametersX['SIL'])+','+str(self.windowParametersX['SIU'])+' ] -->')+self.c.yellow(' [ '+str(self.dataParametersX['SIL'])+','+str(self.dataParametersX['SIU'])+' ]'))
		print('Physical-to-image pedestal integration window map: '+self.c.cyan('[ '+str(self.windowParametersX['PIL'])+','+str(self.windowParametersX['PIU'])+' ] -->')+self.c.yellow(' [ '+str(self.dataParametersX['PIL'])+','+str(self.dataParametersX['PIU'])+' ]'))
		print('                             --and--                             ')
		print('The x-axis range will be: '+self.c.blue('( '+str(self.windowParametersX['xRange'][0])+', '+str(self.windowParametersX['xRange'][1])+' )')+', with ticks at: ')
		print(self.windowParametersX['xTicks'])
		print('\n#################################################################\n')
		
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
		
	def initializeData(self, channels=2):
		
		try:
			print('Making working directory: '+self.c.yellow(str(self.workingDir))+'...')
			subprocess.call('mkdir -p '+str(self.workingDir), shell=True)
		except NameError:
			sys.exit(self.c.red('WorkingDir variable not found. Did you setupClassVariables?'))
		
		if self.load == True:
			print('Loading in trace list...')
			traceList = pd.read_hdf(str(self.loadFrom), 'table')
		else:
			print('Reading in lines... (this takes a lot of memory!)')
			if channels == 2:
				lines = self.openDualRawTraceFile(self.channel1, self.channel2)
				traceList = self.dualTraceToPandas(self.extractDualRawTraces(lines), self.windowParametersX['x'])
				del lines
			elif channels == 1:
				lines = self.openRawTraceFile(self.channel1)
				traceList = self.tracetoPandas(self.extractRawTraces(lines), self.windowParametersX['x'])
				del lines
			if self.saveData == True:
				traceList.to_hdf(str(self.saveTo), 'table')
				
		return traceList
	
		
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
			#returnData.append(self.packageTrace(channel1Data[i],channel2Data[i]))
			returnData.append(np.column_stack((channel1Data[i],channel2Data[i])))
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
		
		returnData = traceData.copy(deep=True)
		
		returnData[returnData.columns[0::2]] = returnData[returnData.columns[0::2]].subtract(pedestalOffsets[0], axis='columns')
		returnData[returnData.columns[1::2]] = returnData[returnData.columns[1::2]].subtract(pedestalOffsets[1], axis='columns')
		
		return returnData
		
		
	##########
	
	#Simple spike rejection
	def spikeRejection(self, traceData, limits, voltageThreshold, timeThreshold, saveSpikes=False):
		
		k=0
		j=0
		returnData = None
		savedSpikes = None
		
		
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
		#print(str(len(returnData.columns[0::2]))+' accepted, '+str(len(savedSpikes.columns[0::2]))+' rejected')
		
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
				dataFrameReturned = pd.DataFrame(d, index=indexArray)
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
		
		

		
		
	def doubleRejection(self, CsITraces, windowParametersX, dataParametersX, SGWindow, SGOrder,
		minimaWindowDR, medianFactorDR, fitWindow, alphaThreshold, chatter=False):
		
		def func(x, alpha, beta, gamma):
			return alpha*x**2 + beta*x + gamma
		
		good = np.array([], dtype=np.dtype(int))
		bad = np.array([], dtype=np.dtype(int))
		
		for i, thisTrace in enumerate(CsITraces.columns):
			if ((i % 100) == 0):
				print('Working on traces '+self.c.blue(str(i)+' - '+str(i+99))+' for double rejection...')
			parameters = np.array([])
			CsITrace = np.array(CsITraces[thisTrace])
			CsISmoothed = signal.savgol_filter(CsITrace, SGWindow, SGOrder, deriv=0)
			minInds = signal.argrelmin(CsISmoothed, order=minimaWindowDR)[0]
			#medianCutoff = -1*medianFactorDR*np.median(abs(CsITrace[dataParametersX['PIL']:dataParametersX['PIU']]))
			medianCutoff = medianFactorDR*abs(CsITraces[windowParametersX['PIL']:windowParametersX['PIU']]).median(axis=1).median()
			medInds = minInds[np.where(CsISmoothed[minInds] < -1*medianCutoff)]
			for j, index in enumerate(medInds):
				try:
					lowerLimit = index-fitWindow
					xLowValue = windowParametersX['x'][lowerLimit]
					del xLowValue
				except IndexError:
					print(self.c.orange('Too close to lower edge, fitting index set to zero.'))
					lowerLimit = 0					
				try:
					upperLimit = index+fitWindow
					xUpperValue = windowParametersX['x'][upperLimit]
					del xUpperValue
				except IndexError:
					print(self.c.orange('Too close to upper edge, fitting index set to array maximum.'))
					upperLimit = len(CsISmoothed)-1
				xdata = windowParametersX['x'][lowerLimit:upperLimit]
				ydata = CsISmoothed[lowerLimit:upperLimit]
				popt, pcov = curve_fit(func, xdata, ydata)
				if popt[0] > alphaThreshold:
					parameters = np.append(parameters, popt[0])

#				try:
#					xdata = windowParametersX['x'][index-250:index+250]
#				except IndexError:
#					print('Too close to edge, skipping...')
#					continue
#				ydata = CsISmoothed[index-250:index+250]
#				popt, pcov = curve_fit(func, xdata, ydata)
#				if popt[0] > 0.035:
#					parameters = np.append(parameters, popt[0])

			if len(parameters) > 1:
				bad = np.append(bad, i)
			else:
				good = np.append(good, i)
									
		acceptedTraces = CsITraces.drop(CsITraces.columns[bad], axis=1).copy(deep=True)
		rejectedTraces = CsITraces.drop(CsITraces.columns[good], axis=1).copy(deep=True)
		print('Traces dropped: '+self.c.orange(str(len(bad))))		
		print('Traces accepted: '+self.c.lightgreen(str(len(good))))
		print('Traces total: '+self.c.lightblue(str(len(good)+len(bad))))
		return acceptedTraces, rejectedTraces, good, bad
			
			
			
	def peakFinder(self, WLSTraces, windowParametersX, dataParametersX, medianFactorPF, stdFactor,
		convWindow, convPower, convSig, minimaWindowPF):

		countedPhotons = np.array([])
		photonInd = []
		print('Looking at '+str(len(WLSTraces.columns))+'traces.')
		for i, thisTrace in enumerate(WLSTraces.columns):
			data = np.array(-1*WLSTraces[thisTrace][windowParametersX['SIL']:windowParametersX['SIU']])
			medCutoff = medianFactorPF*np.median(abs(data))
			stdCutoff = stdFactor*np.std(data[np.where(abs(data) < medCutoff)])
			window = signal.general_gaussian(convWindow, p=convPower, sig=convSig)
			filtered = signal.fftconvolve(window, data)
			filtered = (np.average(data) / np.average(filtered)) * filtered
			filtered = np.roll(filtered, -1*int((convWindow-1)/2))
			peakind = signal.argrelmax(filtered, order=minimaWindowPF)[0]
			newind = peakind[np.where(filtered[peakind] > stdCutoff)]
			photonInd.append(newind)
			countedPhotons = np.append(countedPhotons, len(newind))
			if ((i % 100) == 0):
				print('Working on traces '+self.c.blue(str(i)+' - '+str(i+99))+' for peak detection...')
		print('Total photons counted: '+self.c.lightgreen(str(sum(countedPhotons))))
		print('Average number of photons per trace: '+self.c.lightgreen(str(sum(countedPhotons)/len(WLSTraces.columns))))
		print('Median number of photons: '+self.c.lightgreen(str(np.median(countedPhotons))))
		
		return photonInd, countedPhotons

	def pedestalPlot(self, pedSum, windowParametersX, windowParametersY, legend=None, color=None, 
		xLabel=None, yLabel=None, title=None, lowerLim=None, upperLim=None, number=None, 
		myFont=None, fileName=None, show=False, save=None):
	
		if not legend:
			legend = windowParametersY['object']
		if not color:
			color = windowParametersY['color']
		if not xLabel:
			xLabel = 'Number of Events [$N$]'
		if not yLabel:
			yLabel = 'Summed Voltage [$V$]'
		if not title:
			title = ( 'Summed Pedestal Distribution (Interval: ['+str(windowParametersX['PIL'])+' '
			+windowParametersX['xWidthUnit']+', '+str(windowParametersX['PIU'])+' '
			+windowParametersX['xWidthUnit']+'])' )
		if not lowerLim:
			lowerLim = pedSum.min()
		if not upperLim:
			#upperLim = 3*pedSum.median()
			upperLim = pedSum.max()
		if not number:
			number = int(round(sqrt(pedSum.index.shape[0])))
		if not myFont:
			myFont = {'fontname':'Liberation Serif'}
		if not fileName:
			fileName = 'pedestal.png'
			
		fig, ax = plt.subplots(figsize=(9,6), dpi=100)
		bins = np.linspace(lowerLim, upperLim, number)
		plt.hist(pedSum, bins, label=legend, color=color)
		ax.set_ylabel(xLabel, **myFont)
		ax.set_xlabel(yLabel, **myFont)
		plt.title(title, **myFont)
		plt.legend(loc='upper right')
				

		if save == True:
			plt.savefig(fileName,dpi=500)
		if show == True:
			plt.show()
		plt.close()
		
	
	def pedestalDualPlot(self, pedSum, windowParametersX, windowParametersY1, windowParametersY2, 
		legend1=None, legend2=None, color1=None, color2=None, xLabel=None, yLabel=None, title=None, 
		lowerLim=None, upperLim=None, number=None, myFont=None, fileName=None, show=False, save=None):

		if not legend1:
			legend1 = windowParametersY1['object']
		if not legend2:
			legend2 = windowParametersY2['object']
		if not color1:
			color1 = windowParametersY1['color']
		if not color2:
			color2 = windowParametersY2['color']		
		if not xLabel:
			xLabel = 'Number of Events [$N$]'		
		if not yLabel:
			yLabel = 'Summed Voltage [$V$]'
		if not title:
			title = ('Summed Pedestal Distributions (Interval: ['+str(windowParametersX['PIL'])+' '
				+windowParametersX['xWidthUnit']+', '+str(windowParametersX['PIU'])+' '
				+windowParametersX['xWidthUnit']+'])')
		if not lowerLim:
			lowerLim = pedSum.min()
		if not upperLim:
			upperLim = np.max([3*pedSum[0::2].median(),3*pedSum[1::2].median()])
		if not number:
			number = int(round(2*sqrt(pedSum.index.shape[0])))
		if not myFont:
			myFont = {'fontname':'Liberation Serif'}			
		if not fileName:
			fileName = 'pedestals.png'				
		
		fig, ax = plt.subplots(figsize=(9,6), dpi=100)
		ax.set_ylabel(xLabel, **myFont)
		ax.set_xlabel(yLabel, **myFont)
		plt.title(title, **myFont)
		plt.legend(loc='upper right')

		bins = np.linspace(lowerLim, upperLim, number)
		plt.hist(pedSum[0::2], bins, label=legend1, color=color1)
		plt.hist(pedSum[1::2], bins, label=legend2, color=color2)
		
		if save == True:
			plt.savefig(fileName,dpi=500)
		if show == True:
			plt.show()
		plt.close()

		
	def plotPHD(self, sums, windowParametersX, windowParametersY, legend=None, color=None,
		xLabel=None, yLabel=None, title=None, bins=None, ylim=None, myFont=None, fileName=None, show=False, save=None):
		
		if not legend:
			legend = windowParametersY['object']
		if not color:
			color = windowParametersY['color']
		if not xLabel:
			xLabel = 'Number of Events [$N$]'
		if not yLabel:
			yLabel = 'Summed Voltage [$V$]'
		if not title:
			title = ( windowParametersY['object']+' summed signal distribution (Interval: ['
				+str(windowParametersX['SIL'])+' '+windowParametersX['xWidthUnit']+', '
				+str(windowParametersX['SIU'])+' '+windowParametersX['xWidthUnit']+'])' )
		if not np.any(bins):
			#print('Bins for '+windowParametersY['object']+' will be: ')
			bins = np.linspace(np.min(sums),np.max(sums), int(round(2*sqrt(sums.index.shape[0]))))
			print(bins)
		if not myFont:
			myFont = {'fontname':'Liberation Serif'}
		if not fileName:
			fileName = 'pedestal.png'		

		fig, ax = plt.subplots(figsize=(9,6), dpi=100)
		ax.set_ylabel(xLabel, **myFont)
		ax.set_xlabel(yLabel, **myFont)
		ax.set_title(title, **myFont)
		ax.legend(loc='upper right')
		
		plt.hist(sums, bins, label=legend, color=color)
		if ylim:
			ax.set_ylim(0, ylim)
		if save == True:	
			plt.savefig(fileName,dpi=500)
		if show == True:
			plt.show()
		plt.close()
		
		return bins

	def plotTrace(self, trace, windowParametersX, windowParametersY, legend=None,
		color=None, xLabel=None, yLabel=None, title=None, myFont=None, fileName=None):
		
		if not legend:
			legend = windowParametersY['object']
		if not color:
			color = windowParametersY['color']	
		if not xLabel:
			if windowParametersX['selection'] == 'relative':
				xLabel = 'Time Relative to Trigger ['+windowParametersX['xWidthUnit']+']'
			else:
				xLabel = 'Time ['+windowParametersX['xWidthUnit']+']'
		if not yLabel:
			yLabel = windowParametersY['object']+' Voltage [V]'
		if not title:
			title = ('APT Raw Detector Trace')
		if not fileName:
			fileName = 'raw_trace.png'
		if not myFont:
			myFont = {'fontname':'Liberation Serif'}
			
	def plotDualTrace(self, trace1, trace2, windowParametersX, windowParametersY1, windowParametersY2,
		legend1=None, legend2=None, color1=None, color2=None, xLabel=None, yLabel1=None, yLabel2=None,
		title=None , myFont=None, fileName=None, show=None, save=None):
		
		if not legend1:
			legend1 = windowParametersY1['object']
		if not legend2:
			legend2 = windowParametersY2['object']
		if not color1:
			color1 = windowParametersY1['color']
		if not color2:
			color2 = windowParametersY2['color']		
		if not xLabel:
			if windowParametersX['selection'] == 'relative':
				xLabel = 'Time Relative to Trigger ['+windowParametersX['xWidthUnit']+']'
			else:
				xLabel = 'Time ['+windowParametersX['xWidthUnit']+']'
		if not yLabel1:
			yLabel1 = windowParametersY1['object']+' Voltage [V]'
		if not yLabel2:
			yLabel2 = windowParametersY2['object']+' Voltage [V]'
		if not title:
			title = ('APT Raw Detector Trace')
		if not fileName:
			fileName = 'raw_trace.png'
		if not myFont:
			myFont = {'fontname':'Liberation Serif'}
		
		SIL = windowParametersX['SIL']
		SIU = windowParametersX['SIU']
		PIL = windowParametersX['PIL']
		PIU = windowParametersX['PIU']
		VS = np.min([windowParametersY1['yRange'][0], windowParametersY2['yRange'][0]])
		VE = np.max([windowParametersY1['yRange'][1], windowParametersY2['yRange'][1]])		
		
		fig, ax1 = plt.subplots(figsize=(9,6), dpi=100)
		plt.title(title,**myFont)		
		ax1.grid(True)
		ax1.plot(windowParametersX['x'], trace1, color=color1, linewidth=0.5, linestyle="-")
		ax1.set_xlabel(xLabel, **myFont)
		ax1.set_xlim(windowParametersX['xRange'][0], windowParametersX['xRange'][1])
		ax1.set_xticks(windowParametersX['xTicks'])
		ax1.set_ylim(windowParametersY1['yRange'][0], windowParametersY1['yRange'][1])
		ax1.set_ylabel(yLabel1,**myFont)
		ax1.set_yticks(windowParametersY1['yTicks'])
		ax1.tick_params('y',colors=color1)
		
		ax2 = ax1.twinx()
		ax2.plot(windowParametersX['x'], trace2, color=color2, linewidth=0.5, linestyle="-")
		ax2.set_ylim(windowParametersY2['yRange'][0], windowParametersY2['yRange'][1])
		ax2.set_ylabel(yLabel2,**myFont)
		ax2.set_yticks(windowParametersY2['yTicks'])
		ax2.tick_params('y',colors=color2)
		fig.tight_layout()
		
		plt.plot([SIL,SIL],[VS,VE],color='cyan',linestyle="--",alpha=0.65)
		plt.plot([SIU,SIU],[VS,VE],color='cyan',linestyle="--",alpha=0.65)
		plt.plot([PIL,PIL],[VS,VE],color='purple',linestyle="--",alpha=0.65)
		plt.plot([PIU,PIU],[VS,VE],color='purple',linestyle="--",alpha=0.65)
	
		blue_patch = mpatches.Patch(color=color1, label=legend1)
		red_patch = mpatches.Patch(color=color2, label=legend2)
		mpl.rc('font',family='Liberation Serif')		
		plt.legend(loc='lower right',handles=[red_patch,blue_patch])		

		if save == True:
			plt.savefig(fileName,dpi=500)
		if show == True:
			plt.show()
		
		plt.close()
		
		
	def plotPhotons(self, photonFiles, bins=None, ylim=None, labels=None, colors=None, xLabel=None, yLabel=None,
		title=None, myFont=None, filename=None, show=False, save=None):

		CP = []
		handies = []
		fig, ax = plt.subplots(figsize=(9,6), dpi=100)
		for i, ele in enumerate(photonFiles):
			with open(ele,'r') as f:
				lines = [float(line.strip('\n')) for line in f]
				CP.append(lines)
		n = len(CP)

		if not np.any(bins):
			bins = np.arange(0,60)
		if ylim:
			ax.set_ylim(0, ylim)
		if not labels:
			labels = [str(x) for x in list(np.arange(n))]
		if not colors:
			colors = self.photonColors 
		if not xLabel:
			xLabel = 'Number of photons [$N$]'
		if not yLabel:
			yLabel = 'Fraction of total events'
		if not title:
			title = 'Photon arrival with varying source position'
		if not myFont:
			myFont = {'fontname':'Liberation Serif'}
		if not filename:
			filename = 'countedPhotons.png'

		ax.set_ylabel(yLabel, **myFont)
		ax.set_xlabel(xLabel, **myFont)
		ax.set_title(title, **myFont)
		for i, ele in reversed(list(enumerate(CP))):
			plt.hist(CP[i], bins=bins, normed=True, alpha=0.75, histtype='step', lw=1, color=colors[i])
			handies.append(mlines.Line2D([],[],color=colors[i],label=labels[i]))
		ax.legend(handles=handies, loc='upper right',fontsize='x-small')		
		if save == True:
			plt.savefig(filename,dpi=500)
		if show == True:
			plt.show()
		plt.close()
		
		
class colors:
	
	
	
	BLACK = ''
	BLUE = ''
	GREEN = ''
	CYAN = ''
	RED = ''
	PURPLE = ''
	BROWN = ''
	GRAY = ''
	DARKGRAY = ''
	LIGHTBLUE = ''
	LIGHTGREEN = ''
	LIGHTCYAN = ''
	LIGHTRED = ''
	LIGHTPURPLE = ''
	YELLOW = ''
	WHITE = ''
	BOLD = ''
	UNDERLINE = ''
	ENDC = ''
	
	def enableColors(self):

		self.RED = '\033[0;31m'		
		self.ORANGE = '\033[38;5;166m'
		self.YELLOW = '\033[1;33m'
		self.GREEN = '\033[0;32m'
		self.BLUE = '\033[0;34m'
		self.INDIGO = '\033[38;5;53m'
		self.VIOLET = '\033[38;5;163m'	
		self.PINK =  '\033[38;5;205m'
		self.BLACK = '\033[0;30m'
		self.CYAN = '\033[0;36m'
		self.PURPLE = '\033[0;35m'
		self.BROWN = '\033[0;33m'
		self.GRAY = '\033[0;37m'
		self.DARKGRAY = '\033[1;30m'
		self.LIGHTBLUE = '\033[1;34m'
		self.LIGHTGREEN = '\033[1;32m'
		self.LIGHTCYAN = '\033[1;36m'
		self.LIGHTRED = '\033[1;31m'
		self.LIGHTPURPLE = '\033[1;35m'
		self.WHITE = '\033[1;37m'
		self.BOLD = '\033[1m'
		self.UNDERLINE = '\033[4m'
		self.ENDC = '\033[0m'

		
	def disableColors(self):
		
		self.RED = ''		
		self.ORANGE = ''
		self.YELLOW = ''
		self.GREEN = ''
		self.BLUE = ''
		self.INDIGO = ''
		self.VIOLET = ''
		self.PINK = ''																		
		self.BLACK = ''
		self.CYAN = ''
		self.PURPLE = ''
		self.BROWN = ''
		self.GRAY = ''
		self.DARKGRAY = ''
		self.LIGHTBLUE = ''
		self.LIGHTGREEN = ''
		self.LIGHTCYAN = ''
		self.LIGHTRED = ''
		self.LIGHTPURPLE = ''
		self.WHITE = ''
		self.BOLD = ''
		self.UNDERLINE = ''
		self.ENDC = ''
		
	def getState(self):
		if self.ENDC:
			return True
		elif not self.ENDC:
			return False
		else:
			return -1
			
	def flipState(self):
		if self.getState():
			self.disableColors()
		elif not self.getState():
			self.enableColors()
		else:
			sys.exit("Can't flip ANSI state, exiting.")
			
	def confirmColors(self):
		if self.getState() ==  True:		
			print('Colors are '+self.red('e')+self.orange('n')+self.yellow('a')+self.green('b')+self.blue('l')+self.indigo('e')+self.violet('d'))
		elif self.getState() == False:
			print('Colors are off!')
		elif self.getState() == -1:
			print('Error: Can\'t get color state.')
			
	def confirmColorsDonger(self):
		if self.getState() ==  True:
			print('Colors are '+self.pink('(ﾉ')+self.lightblue('◕')+self.pink('ヮ')+self.lightblue('◕')+self.pink('ﾉ')+self.red('☆')+self.orange('.')+self.yellow('*')+self.green(':')+self.blue('･ﾟ')+self.indigo('✧')+self.violet(' enabled!'))	
		elif self.getState() == False:
			print('Colors are off!')
		elif self.getState() == -1:
			print('Error: Can\'t get color state.')
			
	def orange(self, inString):
		inString = str(self.ORANGE+str(inString)+self.ENDC)
		return inString
	def indigo(self, inString):
		inString = str(self.INDIGO+str(inString)+self.ENDC)
		return inString
	def violet(self, inString):
		inString = str(self.VIOLET+str(inString)+self.ENDC)
		return inString
	def pink(self, inString):
		inString = str(self.PINK+str(inString)+self.ENDC)
		return inString
	def black(self, inString):
		inString = str(self.BLACK+str(inString)+self.ENDC)
		return inString
	def blue(self, inString):
		inString = str(self.BLUE+str(inString)+self.ENDC)
		return inString
	def green(self, inString):
		inString = str(self.GREEN+str(inString)+self.ENDC)
		return inString
	def cyan(self, inString):
		inString = str(self.CYAN+str(inString)+self.ENDC)
		return inString
	def red(self, inString):
		inString = str(self.RED+str(inString)+self.ENDC)
		return inString
	def purple(self, inString):
		inString = str(self.PURPLE+str(inString)+self.ENDC)
		return inString
	def brown(self, inString):
		inString = str(self.BROWN+str(inString)+self.ENDC)
		return inString
	def gray(self, inString):
		inString = str(self.GRAY+str(inString)+self.ENDC)
		return inString
	def darkgray(self, inString):
		inString = str(self.DARKGRAY+str(inString)+self.ENDC)
		return inString
	def lightblue(self, inString):
		inString = str(self.LIGHTBLUE+str(inString)+self.ENDC)
		return inString
	def lightgreen(self, inString):
		inString = str(self.LIGHTGREEN+str(inString)+self.ENDC)
		return inString
	def lightcyan(self, inString):
		inString = str(self.LIGHTCYAN+str(inString)+self.ENDC)
		return inString
	def lightred(self, inString):
		inString = str(self.LIGHTRED+str(inString)+self.ENDC)
		return inString
	def yellow(self, inString):
		inString = str(self.YELLOW+str(inString)+self.ENDC)
		return inString
	def white(self, inString):
		inString = str(self.WHITE+str(inString)+self.ENDC)
		return inString
	def bold(self, inString):
		inString = str(self.BOLD+str(inString)+self.ENDC)
		return inString
	def underline(self, inString):
		inString = str(self.UNDERLINE+str(inString)+self.ENDC)
		return inString
