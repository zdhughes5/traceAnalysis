###########################
"""traceAnalysis Changes"""
###########################

Date: 8/1/17
"traceHandler"
	line 233: changed "peakFinder" to "PeakFinder"

Date: 8/2/17
"traceExtractor"
	line 36: moved "if te.doPhotonCounting" statement to line 36
	line 37: added descriptor print('Graphing photon counts...')
	line 40: changed directory for saving allPhotons.png graph from 'plotsDir' to 'workingDir'
	lines 41-182: added rest of code to else statement to read in data if doPhotonCounting==False
	lines 33-34: moved "te=" and "te.setupClassVariables()" lines to lines 33 and 34, since they are necessary for photon counting as well
	line 32: added descriptor print('Setting up variables...') at line 32
"traceHandler"
	line 301: added colors '#42f48c', '#5909ed', '#e59409', and '#492500' to defaultColors
	line 886: added "ele = ele.strip()" to strip out any accidental white spaces in photonFiles list in the meta.config file

Date: 8/3/17
"traceExtractor"
	line 39: changed title of allPhotons.png graph from "Photon arrival with varying trigger level" to "Photon arrival with varying translator x-positions"

Date: 8/7/17
"traceExtractor"
	line 39: changed user-set "labels=" to "labels=te.photonLabels" and "tite=" to "title=te.photonTitle" so user does not have to edit the traceExtractor.py file
"traceHandler"
	line 246: added "self.photonLabels = self.config['PhotonCounting']['photonLabels'].split(',')" so user does not have to edit the traceExtractor.py file to edit the photon counting graph labels
	line 247: added "self.photonTitle = self.config['PhotonCounting']['photonTitle']" so user does not have to edit the traceExtractor.py file to edit the photon counting graph title
"master_template.conf"
	line 79: added line (and comment) for user to define photon counting graph lables in config file	
	line 80: added line for user to define photon counting graph title in config file
