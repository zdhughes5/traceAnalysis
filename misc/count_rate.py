import matplotlib.pyplot as plt
import numpy as np

def extractData(fileName):
	
	with open(fileName, 'r') as f:
		next(f)
		next(f)
		lines = [x.strip('\n') for x in f]
										
	v, t, c = zip(*[ (float(line.split()[0]), float(line.split()[1]), float(line.split()[2])) for line in lines])

	list(v)
	list(t)
	list(c) 

	r = [c/t for c,t in zip(c,t)]
	
	return (v, t, c , r)
	
def rateFraction(bg, total):
	
	f = [bg/total for bg,total in zip(bg,total)]

	return f
	
fileNameSource100 = '/nfs/data_disks/herc0a/collaborations/APT/data/count_rates/source_CR_x100.dat'
fileNameSource50 = '/nfs/data_disks/herc0a/collaborations/APT/data/count_rates/source_CR_x50.dat'
fileNameSource20 = '/nfs/data_disks/herc0a/collaborations/APT/data/count_rates/source_CR_x20.dat'
fileNameSource10 = '/nfs/data_disks/herc0a/collaborations/APT/data/count_rates/source_CR_x10.dat'

fileNameBg100 = '/nfs/data_disks/herc0a/collaborations/APT/data/count_rates/bg_CR_x100.dat'
fileNameBg50 = '/nfs/data_disks/herc0a/collaborations/APT/data/count_rates/bg_CR_x50.dat'
fileNameBg20 = '/nfs/data_disks/herc0a/collaborations/APT/data/count_rates/bg_CR_x20.dat'
fileNameBg10 = '/nfs/data_disks/herc0a/collaborations/APT/data/count_rates/bg_CR_x10.dat'


vSource100, tSource100, cSource100, rSource100 = extractData(fileNameSource100) 
vSource50, tSource50, cSource50, rSource50 = extractData(fileNameSource50) 
vSource20, tSource20, cSource20, rSource20 = extractData(fileNameSource20)
vSource10, tSource10, cSource10, rSource10 = extractData(fileNameSource10) 

vBg100, tBg100, cBg100, rBg100 = extractData(fileNameBg100) 
vBg50, tBg50, cBg50, rBg50 = extractData(fileNameBg50) 
vBg20, tBg20, cBg20, rBg20 = extractData(fileNameBg20) 
vBg10, tBg10, cBg10, rBg10 = extractData(fileNameBg10) 


plt.figure(1)

plt.plot(vSource100,rSource100,'ro', label='x100 Gain')
plt.plot(vSource100,rSource100,'r')

plt.plot(vSource50,rSource50,'bo', label='x50 Gain')
plt.plot(vSource50, rSource50,'b')

plt.plot(vSource20,rSource20,'co', label='x20 Gain')
plt.plot(vSource20,rSource20,'c')

plt.plot(vSource10,rSource10,'go', label='x10 Gain')
plt.plot(vSource10,rSource10,'g')

myFont = {'fontname':'Liberation Serif'}
plt.title('Source+BG Count Rate vs Discriminator Threshold',**myFont)
plt.ylabel('Count rate [N/s]',**myFont)
plt.xlabel('Threshold level [V]',**myFont)
plt.xticks(np.linspace(0,10,11))
plt.yticks(np.linspace(0,50000,11))
plt.grid(True)
plt.legend(loc='upper right')


plt.figure(2)

plt.plot(vBg100,rBg100,'ro', label='x100 Gain')
plt.plot(vBg100,rBg100,'r')

plt.plot(vBg50,rBg50,'bo', label='x50 Gain')
plt.plot(vBg50, rBg50,'b')

plt.plot(vBg20,rBg20,'co', label='x20 Gain')
plt.plot(vBg20,rBg20,'c')

plt.plot(vBg10,rBg10,'go', label='x10 Gain')
plt.plot(vBg10,rBg10,'g')

myFont = {'fontname':'Liberation Serif'}
plt.title('BG Count Rate vs Discriminator Threshold',**myFont)
plt.ylabel('Count rate [N/s]',**myFont)
plt.xlabel('Threshold level [V]',**myFont)
plt.xticks(np.linspace(0,10,11))
plt.yticks(np.linspace(0,100,11))
plt.ylim(0,100)
plt.grid(True)
plt.legend(loc='upper right')


plt.figure(3)

rSource100 = rSource100[0:29]
rSource50 = rSource50[0:29]
rSource20 = rSource20[0:29]
rSource10 = rSource10[0:29]

rBg100 = rBg100[0:29]
rBg50 = rBg50[0:29]
rBg20 = rBg20[0:29]
rBg10 = rBg10[0:29]

vSource100 = vSource100[0:29]
vSource50 = vSource50[0:29]
vSource20 = vSource20[0:29]
vSource10 = vSource10[0:29]

f100 = rateFraction(rBg100,rSource100)
f50 = rateFraction(rBg50,rSource50)
f20 = rateFraction(rBg20,rSource20)
f10 = rateFraction(rBg10,rSource10)

plt.plot(vSource100,f100,'ro', label='x100 Gain')
plt.plot(vSource100,f100,'r')

plt.plot(vSource50,f50,'bo', label='x50 Gain')
plt.plot(vSource50,f50,'b')

plt.plot(vSource20,f20,'co', label='x20 Gain')
plt.plot(vSource20,f20,'c')

#plt.plot(vSource10,f10,'go', label='x10 Gain')
#plt.plot(vSource10,f10,'g')

myFont = {'fontname':'Liberation Serif'}
plt.title('BG rate as fraction of total rate',**myFont)
plt.ylabel('Count rate fraction',**myFont)
plt.xlabel('Threshold level [V]',**myFont)
plt.xticks(np.linspace(0,10,11))
plt.yticks(np.linspace(0,0.025,11))
plt.grid(True)
plt.legend(loc='upper right')


plt.show()