import numpy as np
import code
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
from math import sqrt
from scipy import signal
from scipy.optimize import curve_fit


#Parses the lines of the oscilliscope file and pulls out just the trace data. Optional Save.
def extractRawTraces(lines, saveTraces=False, traceFilename = 'trace_'):
	
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
def extractDualRawTraces(lines, saveTraces=False, traceFilename = 'trace_'):
	
	channel1Data = extractRawTraces(lines[0])
	channel2Data = extractRawTraces(lines[1])
	returnData = []
	
	for i, element in enumerate(channel1Data):
		#returnData.append(self.packageTrace(channel1Data[i],channel2Data[i]))
		returnData.append(np.column_stack((channel1Data[i],channel2Data[i])))
		if saveTraces == True:
			np.savetext(traceFilename+str(i)+returnData[i]+'.txt',fmt='%.6e')
	
	return returnData

	
##########	


#Get subtrace based on limits
def extractSubtraces(traceData, limit, invert=False):
	
	if invert==False:
		return traceData[:][limit[0]:limit[1]]
	else:
		return -1*traceData[:][limit[0]:limit[1]]
		
def extractDualSubtraces(traceData, limits, invert=False):
	
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
def sumTraces(traceData):
	
	return traceData[:].sum()
	

##########	


#Get the average of the median value
def getAvgMedian(traceData, intervalSize, invert = False):
	
	if invert == False:
		return traceData.median()/intervalSize
	else:
		return -1*traceData.median()/intervalSize
		
def getDualAvgMedian(traceData, intervalSizes, invert = False):
	
	if invert == False:
		return (traceData[0::2].median()/intervalSizes[0], traceData[1::2].median()/intervalSizes[1])
	else:
		return (-1*traceData[0::2].median()/intervalSizes[0], -1*traceData[1::2].median()/intervalSizes[1])
		
		
##########

#Correct for pedestal offsets
def pedestalDualSubtractions(traceData, pedestalOffsets):
	
	returnData = traceData.copy(deep=True)
	
	returnData[returnData.columns[0::2]] = returnData[returnData.columns[0::2]].subtract(pedestalOffsets[0], axis='columns')
	returnData[returnData.columns[1::2]] = returnData[returnData.columns[1::2]].subtract(pedestalOffsets[1], axis='columns')
	
	return returnData
	
	
##########

#Simple spike rejection
def spikeRejection(traceData, limits, voltageThreshold, timeThreshold, saveSpikes=False):
	
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
def tracetoPandas(traceList, indexArray):
	
	label = 'channel1_'
	
	for i, element in enumerate(traceList):
		if i == 0:
			d = {label+str(i):element[:]}
			dataFrameReturned = pd.DataFrame(d, index=indexArray)
		else:
			dataFrameReturned[label+str(i)] = element[:]

	return dataFrameReturned

def dualTraceToPandas(traceList, indexArray):

	label1 = 'channel1_'
	label2 = 'channel2_'
	
	for i, element in enumerate(traceList):
		if i == 0:
			d = {label1+str(i):element[:,0], label2+str(i):element[:,1]}
			dataFrameReturned = pd.DataFrame(d,index=indexArray)				
		else:
			dataFrameReturned[label1+str(i)], dataFrameReturned[label2+str(i)] = [element[:,0],element[:,1]]
			
	return dataFrameReturned
	
	

	
	
def doubleRejection(CsITraces, windowParametersX, dataParametersX, SGWindow, SGOrder,
	minimaWindowDR, medianFactorDR, fitWindow, alphaThreshold, chatter=False, c=None):
	
	if not c:
		from traceAnalysis.traceHandler import colors
		c = colors()
	
	def func(x, alpha, beta, gamma):
		return alpha*x**2 + beta*x + gamma
	
	good = np.array([], dtype=np.dtype(int))
	bad = np.array([], dtype=np.dtype(int))
	
	for i, thisTrace in enumerate(CsITraces.columns):
		if ((i % 100) == 0):
			print('Working on traces '+c.blue(str(i)+' - '+str(i+99))+' for double rejection...')
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
				print(c.orange('Too close to lower edge, fitting index set to zero.'))
				lowerLimit = 0					
			try:
				upperLimit = index+fitWindow
				xUpperValue = windowParametersX['x'][upperLimit]
				del xUpperValue
			except IndexError:
				print(c.orange('Too close to upper edge, fitting index set to array maximum.'))
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
	print('Traces dropped: '+c.orange(str(len(bad))))		
	print('Traces accepted: '+c.lightgreen(str(len(good))))
	print('Traces total: '+c.lightblue(str(len(good)+len(bad))))
	return acceptedTraces, rejectedTraces, good, bad
		
		
		
def peakFinder(WLSTraces, windowParametersX, dataParametersX, medianFactorPF, stdFactor,
	convWindow, convPower, convSig, minimaWindowPF, plotRaw=None, traceDir=None, 
	showPlots=None, savePlots=None, c = None):
	
	if not c:
		from traceAnalysis.traceHandler import colors
		c = colors()

	countedPhotons = np.array([])
	photonInds = []
	
	print('Looking at '+str(len(WLSTraces.columns))+' traces.')
	for i, thisTrace in enumerate(WLSTraces.columns):
		signalData = np.array(-1*WLSTraces[thisTrace][windowParametersX['SIL']:windowParametersX['SIU']])
		medCutoff = medianFactorPF*np.median(abs(signalData))
		stdCutoff = stdFactor*np.std(signalData[np.where(abs(signalData) < medCutoff)])
		window = signal.general_gaussian(convWindow, p=convPower, sig=convSig)
		smoothedData = signal.fftconvolve(window, signalData)
		smoothedData = (np.average(signalData) / np.average(smoothedData)) * smoothedData
		smoothedData = np.roll(smoothedData, -1*int((convWindow-1)/2))
		peakInds = signal.argrelmax(smoothedData, order=minimaWindowPF)[0]
		cutoffInds = peakInds[np.where(smoothedData[peakInds] > stdCutoff)]

		if (plotRaw is not None and i in plotRaw) or plotRaw=='all':
			width = len(signalData)
			filename = traceDir/str('raw_trace_'+str(i)+'.png')
			f, ax = plt.subplots(figsize=(9,6))
			ax.set_xlabel('Array Index')
			ax.set_xlim([0, len(smoothedData)])
			ax.set_ylabel('-1*Voltage')
			ax.set_title('Peak Finder results with signal integration region.')
			ax.plot(smoothedData, 'blue')
			ax.plot(signalData, color='black', alpha=0.5)
			ax.plot(cutoffInds, smoothedData[cutoffInds],'r.')
			ax.plot([0,width],[medCutoff, medCutoff], linestyle=':')
			ax.plot([0,width],[stdCutoff, stdCutoff], linestyle='-.')
			if savePlots is not None and traceDir is not None:
				plt.savefig(str(filename))
			if showPlots == True:
				plt.show()

			
		returnedInd = cutoffInds + dataParametersX['SIL']
		photonInds.append(returnedInd)
		countedPhotons = np.append(countedPhotons, len(returnedInd))
		#code.interact(local=locals())
		#sys.exit('Code Break!')
		if ((i % 100) == 0):
			print('Working on traces '+c.blue(str(i)+' - '+str(i+99))+' for peak detection...')
	print('Total photons counted: '+c.lightgreen(str(sum(countedPhotons))))
	print('Average number of photons per trace: '+c.lightgreen(str(sum(countedPhotons)/len(WLSTraces.columns))))
	print('Median number of photons: '+c.lightgreen(str(np.median(countedPhotons))))
	

	
	return photonInds, countedPhotons

def pedestalPlot(pedSum, windowParametersX, windowParametersY, legend=None, color=None, 
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
		
	fig, ax = plt.subplots(figsize=(9,6))
	bins = np.linspace(lowerLim, upperLim, number)
	plt.hist(pedSum, bins, label=legend, color=color)
	ax.set_ylabel(xLabel, **myFont)
	ax.set_xlabel(yLabel, **myFont)
	plt.title(title, **myFont)
	plt.legend(loc='upper right')
			

	if save == True:
		plt.savefig(fileName)
	if show == True:
		plt.show()
	plt.close()
	

def pedestalDualPlot(pedSum, windowParametersX, windowParametersY1, windowParametersY2, 
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
	
	fig, ax = plt.subplots(figsize=(9,6))
	ax.set_ylabel(xLabel, **myFont)
	ax.set_xlabel(yLabel, **myFont)
	plt.title(title, **myFont)
	plt.legend(loc='upper right')

	bins = np.linspace(lowerLim, upperLim, number)
	plt.hist(pedSum[0::2], bins, label=legend1, color=color1)
	plt.hist(pedSum[1::2], bins, label=legend2, color=color2)
	
	if save == True:
		plt.savefig(fileName)
	if show == True:
		plt.show()
	plt.close()

	
def plotPHD(sums, windowParametersX, windowParametersY, legend=None, color=None,
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

	fig, ax = plt.subplots(figsize=(9,6))
	ax.set_ylabel(xLabel, **myFont)
	ax.set_xlabel(yLabel, **myFont)
	ax.set_title(title, **myFont)
	ax.legend(loc='upper right')
	
	plt.hist(sums, bins, label=legend, color=color)
	if ylim:
		ax.set_ylim(0, ylim)
	if save == True:	
		plt.savefig(fileName)
	if show == True:
		plt.show()
	plt.close()
	
	return bins

def plotTrace(trace, windowParametersX, windowParametersY, legend=None,
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
		
def plotDualTrace(trace1, trace2, windowParametersX, windowParametersY1, windowParametersY2,
	legend1=None, legend2=None, color1=None, color2=None, xLabel=None, yLabel1=None, yLabel2=None,
	title=None , myFont=None, fileName=None, show=None, save=None, photonInds=None):
	
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
	
	fig, ax1 = plt.subplots(figsize=(9,6))
	plt.title(title,**myFont)		
	ax1.grid(b=True, linestyle=':')
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
	

	
	if legend1 == 'WLS Fiber' and photonInds is not None:
		ax1.scatter(trace1.index.values[photonInds], trace1.iloc[photonInds], marker='+', color='black')
	elif legend2 == 'WLS Fiber' and photonInds is not None:
		ax2.scatter(trace2.index.values[photonInds], trace2.iloc[photonInds], marker='+', color='black')
		
	#code.interact(local=locals())
	#sys.exit('Code Break!')

	blue_patch = mpatches.Patch(color=color1, label=legend1)
	red_patch = mpatches.Patch(color=color2, label=legend2)
	mpl.rc('font',family='Liberation Serif')		
	plt.legend(loc='lower right',handles=[red_patch,blue_patch])		

	if save == True:
		plt.savefig(fileName)
	if show == True:
		plt.show()
	
	plt.close()	
	
def plotPhotons(photonFiles, bins=None, ylim=None, labels=None, colors=None, xLabel=None, yLabel=None,
	title=None, myFont=None, filename=None, show=False, save=None):

	CP = []
	handies = []
	fig, ax = plt.subplots(figsize=(9,6))
	for i, ele in enumerate(photonFiles):
		ele = ele.strip()
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
		colors = ['black' for x in range(0, len(CP))] 
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
		plt.savefig(filename)
	if show == True:
		plt.show()
	plt.close()