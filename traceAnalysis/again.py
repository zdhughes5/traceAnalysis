#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:33:35 2018

@author: zdhughes
"""

import sys
from PyQt5 import QtWidgets
from testme import Ui_MainWindow
from configparser import ConfigParser, ExtendedInterpolation
import traceHandler


class Window(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		
		self.BrowseButton.clicked.connect(self.pickAFile)
		self.LoadButton.clicked.connect(self.loadAfile)
		
	def pickAFile(self):
		fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '.')
		
		if fname[0]:
			self.FilePathText.setText(fname[0])
			self.LoadButton.setEnabled(True)
			
	def loadAfile(self):
		configFile = self.FilePathText.text()
		config = ConfigParser(interpolation=ExtendedInterpolation(),inline_comment_prefixes=('#'))
		config.read(configFile)
		c = traceHandler.colors()
		if config['General'].getboolean('ansiColors') == True:
			c.enableColors()
		c.confirmColorsDonger()
		te = traceHandler.traceHandler(config, c)
		te.setupClassVariables()
		
		#[General]
		self.workingDirRO.setText(str(te.workingDir))
		self.dataDirRO.setText(str(te.dataDir))
		self.plotsDirRO.setText(str(te.plotsDir))
		self.traceDirRO.setText(str(te.traceDir))
		self.ansiColorsRO.setText(str(te.ansiColors))
		self.doPlotsRO.setText(str(te.doPlots))
		
		#[IO]
		self.saveDataRO.setText(str(te.workingDir))
		self.saveToRO.setText(str(te.saveTo))
		self.loadRO.setText(str(te.load))
		self.loadFromRO.setText(str(te.loadFrom))
		self.showPlotsRO.setText(str(te.showPlots))
		self.savePlotsRO.setText(str(te.savePlots))
		self.allPlotsRO.setText(str(te.allPlots))
		
		#[Channels]
		self.doubleChannelRO.setText(str(te.doubleChannel))
		self.BGSubtractionRO.setText(str(te.BGSubtraction))
		self.channel1RO.setText(str(te.channel1))
		self.channel2RO.setText(str(te.channel2))
		self.metaRO.setText(str(te.meta))
		self.channel1BGRO.setText(str(te.channel1BG))
		self.channel2BGRO.setText(str(te.channel2BG))
		self.metaBGRO.setText(str(te.metaBG))
		
		#[Window]
		self.xPlotTypeRO.setText(str(te.xPlotType))
		self.yPlotTypeRO.setText(str(te.yPlotType))
		self.xRelativeGridRO.setText(str(te.xRelativeGrid))
		
		#[Integration]
		self.SILRO.setText(str(te.SIL))
		self.SIURO.setText(str(te.SIU))
		self.PILRO.setText(str(te.PIL))
		self.PIURO.setText(str(te.PIU))
		
		#[SpikeRejection]
		self.doSpikeRejectionRO.setText(str(te.doSpikeRejection))
		self.voltageThresholdRO.setText(str(te.voltageThreshold))
		self.timeThresholdRO.setText(str(te.timeThreshold))
		
		#[SmoothedDoubleRejection]
		self.doDoubleRejectionRO.setText(str(te.doDoubleRejection))
		self.SGWindowRO.setText(str(te.SGWindow))
		self.SGOrderRO.setText(str(te.SGOrder))
		self.minimaWindowDRRO.setText(str(te.minimaWindowDR))
		self.medianFactorDRRO.setText(str(te.medianFactorDR))
		self.fitWindowRO.setText(str(te.fitWindow))
		self.alphaThresholdRO.setText(str(te.alphaThreshold))
		
		#[PeakFinder]
		self.plotRawRO.setText(str(te.plotRaw))
		self.photonFilenameRO.setText(str(te.photonFilename))
		self.doPeakFinderRO.setText(str(te.doPeakFinder))
		self.savePhotonsRO.setText(str(te.savePhotons))
		self.medianFactorPFRO.setText(str(te.medianFactorPF))
		self.stdFactorRO.setText(str(te.stdFactor))
		self.convWindowRO.setText(str(te.convWindow))
		self.convPowerRO.setText(str(te.convPower))
		self.convSigRO.setText(str(te.convSig))
		self.minimaWindowPFRO.setText(str(te.minimaWindowPF))
		
		#[PhotonCounting]
		self.doPhotonCountingRO.setText(str(te.doPhotonCounting))
		self.photonFilesRO.setText(str(te.photonFiles))
		self.photonLabelsRO.setText(str(te.photonLabels))
		self.photonTitleRO.setText(str(te.photonTitle))
		
		#Parse meta file		
		#[General]
		self.xWidthPhysicalRO.setText(str(te.xWidthPhysical))
		self.xWidthUnitRO.setText(str(te.xWidthUnit))
		self.xLocationRO.setText(str(te.xLocation))
		self.yHeightUnitsRO.setText(str(te.yHeightUnits))
		self.triggerRO.setText(str(te.trigger))
		self.triggerSourceRO.setText(str(te.triggerSource))
		self.triggerTypeRO.setText(str(te.triggerType))
		self.sampleRO.setText(str(te.sample))
		self.xDivsRO.setText(str(te.xDivs))
		self.yDivsRO.setText(str(te.yDivs))
		
		#[channel1]
		self.object1RO.setText(str(te.object1))
		self.inputImpedence1RO.setText(str(te.inputImpedence1))
		self.coupling1RO.setText(str(te.coupling1))
		self.offset1RO.setText(str(te.offset1))
		self.bandwidth1RO.setText(str(te.bandwidth1))
		self.VoltsPerDiv1RO.setText(str(te.VoltsPerDiv1))
		self.yLocation1RO.setText(str(te.yLocation1))
		
		#[channel2]
		self.object2RO.setText(str(te.object2))
		self.inputImpedence2RO.setText(str(te.inputImpedence2))
		self.coupling2RO.setText(str(te.coupling2))
		self.offset2RO.setText(str(te.offset2))
		self.bandwidth2RO.setText(str(te.bandwidth2))
		self.VoltsPerDiv2RO.setText(str(te.VoltsPerDiv2))
		self.yLocation2RO.setText(str(te.yLocation2))		
		
		
		
		
app = QtWidgets.QApplication(sys.argv)
window = Window()
window.show()

#window = QMainWindow()
#ui = Ui_MainWindow(window)
#ui.setupUi(window)
sys.exit(app.exec_())