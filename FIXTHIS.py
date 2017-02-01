#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 00:14:23 2017

@author: zdhughes
"""


code.interact(local=locals())
sys.exit('Code Break!')


	
	
	
	
		
	def packageTraces(self, traces1, traces2, saveTraces=False, traceFilename = 'trace_'):
		
		packagedTrace = []
		for i, trash in enumerate(traces1):
			
			packagedTrace.append(np.column_stack((traces1[i],traces2[i])))

			if saveTraces == True:
				np.savetxt(traceFilename+str(i)+'.txt',packagedTrace[i],fmt='%.6e')			

		return packagedTrace