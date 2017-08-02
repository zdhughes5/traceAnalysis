###########################
"""traceAnalysis Changes"""
###########################

Date: 8/1/17
traceHandler
	line 233: changed "peakFinder" to "PeakFinder"

Date: 8/2/17
traceExtractor
	line 36: moved "if te.doPhotonCounting" statement to line 36
	line 37: added descriptor print('Graphing photon counts...')
	line 40: changed directory for saving allPhotons.png graph from 'plotsDir' to 'workingDir'
	lines 41-182: added rest of code to else statement to read in data if doPhotonCounting==False
	lines 33-34: moved "te=" and "te.setupClassVariables()" lines to lines 33 and 34, since they are necessary for photon counting as well
	line 32: added descriptor print('Setting up variables...') at line 32

traceHandler
	line 301: added colors '#42f48c', '#5909ed', '#e59409', and '#492500' to defaultColors
	line 886: added "ele = ele.strip()" to strip out any accidental white spaces in photonFiles list in the meta.config file
	
