"""
Created on Mon Feb  4 17:58:27 2019

@author: Il_Ciancio
"""
#SOME FUNCTION ON THE DATABASE


def getValueFromChannel(Client,Channel,ShotNumber):
	print(Channel)
	# import matplotlib.pyplot as plt
	from sdas.core.SDAStime import Date, Time, TimeStamp
	import numpy as np
	structArray = Client.getData(Channel, '0x0000', ShotNumber)
	struct = structArray[0]
	Value = struct.getData()
	#TIME
	tstart = struct.getTStart()
	tend = struct.getTEnd()
    #Calculate the time between samples
	tbs = ( tend.getTimeInMicros() - tstart.getTimeInMicros() )/(len(Value)*1.0)
	#Get the events  associated with this data
	events = struct.get('events')
	tevent = TimeStamp(tstamp=events[0].get('tstamp')) #The delay of the start time relative to the event time
	delay = tstart.getTimeInMicros() - tevent.getTimeInMicros()
	#Finally create the time array
	times = np.linspace(delay,delay+tbs*(len(Value)-1),len(Value))
	# plt.plot(times, Value)
	# plt.xlabel('time Micros')
	# plt.ylabel(Channel)
	# plt.grid(True)
	# plt.show()
	return Value
	
###################

def Bmagnmirnv( Z_filament,R_filament,I_filament,r_mirnv,z_mirnv):
#-------------------------------------------------------------------------#

#---Filament is in the R-Y plane and Magnetic Field is Evaluated -------------#

#-------------at the Mirnv coordinate in the R-Z plane------------------------#

#-------------------------------------------------------------------------#
	import numpy as np
	from numpy import pi
	Zc = Z_filament
	I = I_filament
	turns = 1															 # I just have one filament of one turn
	N = 100  								   							 # No of grids in the coil ( X-Y plane)
	u0 = 4*pi*0.001  													 # [microWb/(A cm)]
	phi = np.linspace(0, 2*pi, N) 									 	 # For describing a circle (coil)
	Rc = R_filament * np.cos(phi) 										 #R-coordinates of the filament
	Yc = R_filament * np.sin(phi) 										 #Y-coordinates of the filament
	
	#PYTHON RETURN ERROR IF I DON'T PREALLOCATE A VECTOR
	#Lets obtain the position vectors from dl to the mirnov 
	#mirnov is localized in the plane (y=0)

	RR = np.zeros(N)
	Rz = np.zeros(N)
	Ry = np.zeros(N)
	dlR = np.zeros(N)
	dly = np.zeros(N)
	for i in range(N-1):
		RR[i] = r_mirnv - 0.5* (Rc[i]+Rc[i+1]) 
		Rz[i] = z_mirnv - Zc
		Ry[i] = -0.5 * (Yc[i]+Yc[i+1])
		dlR[i]= Rc[i+1]-Rc[i]
		dly[i]=Yc[i+1]-Yc[i]
		
	RR[-1] = r_mirnv - 0.5*(Rc[-1]+Rc[0])
	Rz[-1] = z_mirnv - Zc
	Ry[-1] = -0.5 * (Rc[-1] + Rc[0])
	dlR[-1] = -Rc[-1] + Rc[0]
	dly[-1] = -Yc[-1]+Yc[0]
	
	#dl x r
	Rcross = np.multiply(-dly,Rz) #or -dly * Rz
	Ycross = np.multiply(dlR,Rz)
	Zcross = np.multiply(dly,RR) - np.multiply(dlR,Ry)
	R = np.sqrt(np.square(RR) + np.square(Rz) + np.square(Ry)) #OR 	#R = sqrt(RR**2 + Rz**2 + Ry**2) #R = np.sqrt(RR**2 + Rz**2 + Ry**2)
	
	#dB=m0/4pi (Idl x r)/r^2 
	BR1 = np.multiply(np.divide(I*u0, 4*pi*R**3), Rcross)
	Bz1 = np.multiply(np.divide(I*u0, 4*pi*R**3), Zcross)
	By1 = np.multiply(np.divide(I*u0, 4*pi*R**3), Ycross)
	
	#Initialize sum magnetic field to be zero first
	BR = 0
	Bz = 0
	By = 0
	BR = BR + np.sum(BR1)
	Bz = Bz + np.sum(Bz1)
	By = By + np.sum(By1)
	
	BR = BR * turns
	By = By * turns
	Bz = Bz * turns #units=[uWb / cm^2]
	
	vector = [Z_filament - z_mirnv, R_filament - r_mirnv]  #Vector from center of chamber to mirnov center
	unit_vec = np.divide(vector, np.linalg.norm(vector) )  # Unit vector
	norm_vec = [unit_vec[1], -unit_vec[0] ] 			   # Normal vector, coil direction
	Bmirn = np.absolute( BR * unit_vec[1] + Bz * unit_vec[0])
	Bmirn = np.dot([Bz,BR], norm_vec)
	Bmirn = 0.01 * Bmirn 									 #fator de 0.01 pra ter [T] 
	return Bmirn

###################

def ErrorMirnFunc(parameters, Mirnv_B_exp, r_mirnv, z_mirnv):
	import numpy as np
	B_mirnov = np.array([], np.float32)
	Z_filament, R_filament, I_filament  = parameters	
	for i in range(12):
		B_mirnov = np.append(B_mirnov, Bmagnmirnv( Z_filament,R_filament,I_filament,r_mirnv[i],z_mirnv[i] ) )
	
	error = np.sum( np.abs(Mirnv_B_exp - B_mirnov ) )
	return error



###############	
def getDataFromDatabase(ShotNumber):
	from sdas.core.client.SDASClient import SDASClient
	from sdas.core.SDAStime import Date, Time, TimeStamp
	import numpy as np
	import matplotlib.pyplot as plt
	import sys
    
	def StartSdas():
		host = 'baco.ipfn.ist.utl.pt'
		port = 8888
		client = SDASClient(host,port)	
		return client

	#CHANGE SHOT NUMBER HERE
	print("Import data relative to the ShotNumber:" + str(ShotNumber))
	shotnr = ShotNumber	
	client = StartSdas()

	#numerical integrated mirnov coils signals
	mirnv =	[	'MARTE_NODE_IVO3.DataCollection.Channel_129',
				'MARTE_NODE_IVO3.DataCollection.Channel_130',
				'MARTE_NODE_IVO3.DataCollection.Channel_131',
				'MARTE_NODE_IVO3.DataCollection.Channel_132',
				'MARTE_NODE_IVO3.DataCollection.Channel_133',
				'MARTE_NODE_IVO3.DataCollection.Channel_134',
				'MARTE_NODE_IVO3.DataCollection.Channel_135',
				'MARTE_NODE_IVO3.DataCollection.Channel_136',
				'MARTE_NODE_IVO3.DataCollection.Channel_137',
				'MARTE_NODE_IVO3.DataCollection.Channel_138',
				'MARTE_NODE_IVO3.DataCollection.Channel_139',
				'MARTE_NODE_IVO3.DataCollection.Channel_140']


	#numerical integrated mirnov coils signals with offset correction
	mirnv_corr = [	'MARTE_NODE_IVO3.DataCollection.Channel_166',
					'MARTE_NODE_IVO3.DataCollection.Channel_167',
					'MARTE_NODE_IVO3.DataCollection.Channel_168',
					'MARTE_NODE_IVO3.DataCollection.Channel_169',
					'MARTE_NODE_IVO3.DataCollection.Channel_170',
					'MARTE_NODE_IVO3.DataCollection.Channel_171',
					'MARTE_NODE_IVO3.DataCollection.Channel_172',
					'MARTE_NODE_IVO3.DataCollection.Channel_173',
					'MARTE_NODE_IVO3.DataCollection.Channel_174',
					'MARTE_NODE_IVO3.DataCollection.Channel_175',
					'MARTE_NODE_IVO3.DataCollection.Channel_176',
					'MARTE_NODE_IVO3.DataCollection.Channel_177']

	#Calculated external flux on each minov coil through the SS modele
	ext_flux = ['MARTE_NODE_IVO3.DataCollection.Channel_214',
				'MARTE_NODE_IVO3.DataCollection.Channel_215',
				'MARTE_NODE_IVO3.DataCollection.Channel_216',
				'MARTE_NODE_IVO3.DataCollection.Channel_217',
				'MARTE_NODE_IVO3.DataCollection.Channel_218',
				'MARTE_NODE_IVO3.DataCollection.Channel_219',
				'MARTE_NODE_IVO3.DataCollection.Channel_220',
				'MARTE_NODE_IVO3.DataCollection.Channel_221',
				'MARTE_NODE_IVO3.DataCollection.Channel_222',
				'MARTE_NODE_IVO3.DataCollection.Channel_223',
				'MARTE_NODE_IVO3.DataCollection.Channel_224',
				'MARTE_NODE_IVO3.DataCollection.Channel_225']

	#Minov Coils signals with the effect from the external fluxes subtracted		
	mirnv_corr_flux = [	'MARTE_NODE_IVO3.DataCollection.Channel_202',
						'MARTE_NODE_IVO3.DataCollection.Channel_203',
						'MARTE_NODE_IVO3.DataCollection.Channel_204',
						'MARTE_NODE_IVO3.DataCollection.Channel_205',
						'MARTE_NODE_IVO3.DataCollection.Channel_206',
						'MARTE_NODE_IVO3.DataCollection.Channel_207',
						'MARTE_NODE_IVO3.DataCollection.Channel_208',
						'MARTE_NODE_IVO3.DataCollection.Channel_209',
						'MARTE_NODE_IVO3.DataCollection.Channel_210',
						'MARTE_NODE_IVO3.DataCollection.Channel_211',
						'MARTE_NODE_IVO3.DataCollection.Channel_212',
						'MARTE_NODE_IVO3.DataCollection.Channel_213']

	#Measured currents applied by the Primary, Horizontal and Vertical PowerSupply						
	prim = 'MARTE_NODE_IVO3.DataCollection.Channel_093'
	hor = 'MARTE_NODE_IVO3.DataCollection.Channel_091'
	vert = 'MARTE_NODE_IVO3.DataCollection.Channel_092'
    
	#plasma current measured by the Rogowski coil
	Ip_rog = 'MARTE_NODE_IVO3.DataCollection.Channel_088'
	#Ip_rog_value = getValueFromChannel(client,Ip_rog,shotnr)
	
    #chopper =' MARTE_NODE_IVO3.DataCollection.Channel_141'

    #Plasma current reconstructed by the mirnov coils without the correction from external fluxes
	Ip_magn = 'MARTE_NODE_IVO3.DataCollection.Channel_085'
	#Ip_magn_value = getValueFromChannel(client,Ip_magn,shotnr)

    #Plasma current reconstructed by the mirnov coils with the correction from external fluxes
	Ip_magn_corr = 'MARTE_NODE_IVO3.DataCollection.Channel_228'
	Ip_magn_corr_value = getValueFromChannel(client,Ip_magn_corr,shotnr)
	
	#OTHER DATA
	r0_corr_ch='MARTE_NODE_IVO3.DataCollection.Channel_083';
	r0_corr_value = getValueFromChannel(client,r0_corr_ch,shotnr)
	
	z0_corr_ch='MARTE_NODE_IVO3.DataCollection.Channel_084';
	z0_corr_value = getValueFromChannel(client,z0_corr_ch,shotnr)
	
	r0_probes_ch='MARTE_NODE_IVO3.DataCollection.Channel_081';
	r0_probes_value = getValueFromChannel(client,r0_probes_ch,shotnr)
	
	z0_probes_ch='MARTE_NODE_IVO3.DataCollection.Channel_082';
	z0_probes_value = getValueFromChannel(client,z0_probes_ch,shotnr)
	
	sumIfil_ch='MARTE_NODE_IVO3.DataCollection.Channel_230';
	sumIfil_value = getValueFromChannel(client,sumIfil_ch,shotnr)

	#SAVES MIRNOV DATA IN A MATRIX
	coilNr=0
	data_mirnv_corr=[]
	for coil in mirnv_corr:
		coilNr+=1
		structArray=client.getData(coil,'0x0000', shotnr)
		struct=structArray[0]
		data_mirnv_corr.append(struct.getData())

	coilNr=0
	data_mirnv=[]
	for coil in mirnv:
		coilNr+=1
		structArray=client.getData(coil,'0x0000', shotnr)
		struct=structArray[0]
		data_mirnv.append(struct.getData())

	#TIME
	tstart = struct.getTStart()
	tend = struct.getTEnd()
    
	#Calculate the time between samples
	tbs = (tend.getTimeInMicros() - tstart.getTimeInMicros())/(len(data_mirnv_corr[coilNr-1])*1.0)
	
    #Get the events  associated with this data
	events = struct.get('events')
	tevent = TimeStamp(tstamp=events[0].get('tstamp'))
	
    #The delay of the start time relative to the event time
	delay = tstart.getTimeInMicros() - tevent.getTimeInMicros()
	
    #Finally create the time array
	times = np.linspace(delay,delay+tbs*(len(data_mirnv_corr[coilNr-1])-1),len(data_mirnv_corr[coilNr-1]))
    

    # #PLOTS ALL DATA FROM MIRNOVS
	# coilNr=0
	# data[9] = data[9] * 1.2823 #factor
	# for coil in data:
		# coilNr+=1
		# #plt.title('Coil' + str(coilNr))
		# ax = plt.subplot(4, 3, coilNr)
		# ax.set_title('coil'+str(coilNr))
		# plt.plot(times, coil)
		# plt.grid(True)
		
	# plt.show()
	
	#SAVES MIRNOV_FLUX DATA IN A MATRIX
	coilNr=0
	data_mirnv_corr_flux=[]
	for coil in mirnv_corr_flux:
		coilNr+=1
		structArray=client.getData(coil,'0x0000', shotnr)
		struct=structArray[0]
		data_mirnv_corr_flux.append(struct.getData())
	    #PLOTS ALL DATA FROM MIRNOVS
	
	coilNr=0
	
	# for coil in data_flux:
		# coilNr+=1
		# #plt.title('Coil_flux' + str(coilNr))
		# ax = plt.subplot(4, 3, coilNr)
		# ax.set_title('coil_flux'+str(coilNr))
		# plt.plot(times, coil)
		# plt.grid(True)
	#correction of mirnov 10
#	data_mirnv_corr_flux[9] = data_mirnv_corr_flux[9] * 1.2823 #factor
#	data_mirnv[9] = data_mirnv[9] * 1.2823 #factor
#	data_mirnv_corr[9] = data_mirnv_corr[9] * 1.2823 #factor
	# plt.show()
	#data_mirnv IS mirnv
	#data_mirnv_corr_flux IS mirnv_corr_flux
	#data_mirnv_corr is minv_corr
	return [data_mirnv_corr, data_mirnv_corr_flux, Ip_magn_corr_value, times, sumIfil_value, r0_probes_value, z0_probes_value, r0_corr_value, z0_corr_value]

	
###################

def centroidPosition_one_filament(times, Mirnv_flux, Mirnv_flux_corr, SHOT_NUMBER, Ip):
	import numpy as np
	from numpy import pi
	from scipy.optimize import fmin
	import matplotlib.pyplot as plot
	
	time = 1e-03 * times #TIME IN ms
	#DRAW THE VESSEL
	N = 100
	th_vessel = np.linspace(0, 2*pi, N)
	x_vessel = 9 * np.cos(th_vessel) + 46 
	y_vessel = 9 * np.sin(th_vessel)
	
	#MIRNOV POSITION IN DEGREE
	R_mirnov = np.array([], np.float32)
	z_mirnov = np.array([], np.float32)
	ang_mirnov = -15
	for i in range(12):
		R_mirnov = np.append( R_mirnov, 9.35 * np.cos( np.deg2rad(ang_mirnov)  ) + 46 ) 
		z_mirnov = np.append( z_mirnov, 9.35 * np.sin( np.deg2rad(ang_mirnov) ) )
		ang_mirnov = ang_mirnov - 30
	
	# Lets draw the plasma filaments	
	th_filament = np.linspace(0, 2*pi, N)
	R_pls = 46
	z_plsm = 0
	R_filaments = np.array([], np.float32)
	z_filaments = np.array([], np.float32)
	R_filaments = np.append( R_filaments, 46)
	z_filaments = np.append( z_filaments, 0)
	degr_filament = 0
	radius = 4 # in [cm]
	for i in range(1,7) :
		R_filaments = np.append( R_filaments,(46) + radius * np.cos( np.deg2rad(degr_filament) ))
		z_filaments = np.append( z_filaments, radius * np.sin( np.deg2rad(degr_filament) ) )
		degr_filament = degr_filament + 60;

	#EXPERIMENTAL MESUREMENTS [WB], I PRE MULTIPLIED Mirnv_10_fact=1.2823
	time_index = np.where(time == 115)
	
	#Find the exprimental values for that time moment
	
	Mirnov_flux = [i[time_index] for i in Mirnv_flux] #without external flux correction
	
	Mirnov_flux_corr = [i[time_index] for i in Mirnv_flux_corr] #with external flux correction
	
	#Let's go from [Wb] to [T]
	Mirnv_B_exp = np.divide(Mirnov_flux, ( 50 * 49e-6) ) # [T]
	Mirnv_B_exp_corr = np.divide(Mirnov_flux_corr, ( 50 * 49e-6) ) # [T]
	
	#1et's approximation just one filament in the center with 3 degrees of freedom
	
	# Minimization function with and without external flux correction ,
	# units are given in [cm] and [A]
	f_opt = fmin(ErrorMirnFunc,[0,46,4000],args=(Mirnv_B_exp,R_mirnov,z_mirnov), xtol=0.0001, ftol=0.0001)
	f_opt_corr = fmin(ErrorMirnFunc,[0,46,4000],args=(Mirnv_B_exp_corr,R_mirnov,z_mirnov), xtol=0.0001, ftol=0.0001)
	
	#Lets check how close is our minimization values to the experimental ones by applaying Biot-Savart with them 
	Mirnov_flux_experimental = np.array([], np.float32)
	Mirnov_flux_corr_experimental = np.array([], np.float32)
	
	for i in range(12):
		Mirnov_flux_experimental = np.append( Mirnov_flux_experimental, Bmagnmirnv(f_opt[0], f_opt[1], f_opt[2], R_mirnov[i], z_mirnov[i]))
		Mirnov_flux_corr_experimental = np.append( Mirnov_flux_corr_experimental, Bmagnmirnv(f_opt_corr[0], f_opt_corr[1], f_opt_corr[2], R_mirnov[i], z_mirnov[i]))
		
	#compute de error
	RMSE = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_experimental, Mirnv_B_exp ) ) ) )
	RMSE_corr = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_corr_experimental, Mirnv_B_exp_corr ) ) ) )
	#PLOTTIAMO
	plot.figure(1)
	plot.plot(time, Ip, label='Plasma Current')
	plot.plot(time[time_index], Ip[time_index], '.k', MarkerSize=10, label='Chosen Flat-Top')
	plot.legend(loc='upper right')
	plot.xlabel('time[ms]')
	plot.ylabel('I[A]')
	plot.savefig('1_filament_flattopchosen.jpg', dpi=600)
	
	#PLOTTIAMO
	plot.figure(2)
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp, '-o',label="Experimental Data")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_experimental, '-*',label="Biot-savart(optimized )")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA]")
	plot.legend(loc='upper right')
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	plot.savefig('1_filament_RMSE.jpg', dpi=600)
	
	plot.figure(3)
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp_corr, '-o',label="Experimental Data corrected")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_experimental, '-*',label="Biot-savart(optimized )")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (External flux corrected)")
	plot.legend([p1, p2])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	plot.savefig('1_filament_RMSE_correct.jpg', dpi=600)
	
	#Plasma, vessel and mirnov coil plot
	plot.figure(4)
	plot.plot(x_vessel,y_vessel,color='k', linewidth=2.0)
	plot.plot(46,0,'.m',MarkerSize = 480)
	plot.plot(f_opt_corr[1], f_opt_corr[0],'.b',MarkerSize = 20, label = "with data correction If="+str(f_opt_corr[2])+"[A]")
	plot.plot(f_opt[1], f_opt[0],'.r',MarkerSize = 10, label = "without data correction If="+str(f_opt[2])+"[A]")
	for i in range(12):
		plot.text(R_mirnov[i], z_mirnov[i], str(i+1), horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	
	plot.text(57,0,'LFS',FontSize = 15)
	plot.text(33,0,'HFS',FontSize = 15)
	plot.xlabel('R[cm]')
	plot.ylabel('Z[cm]')
	plot.grid()
	plot.legend(loc='upper right')
	plot.axis('equal')
	plot.savefig('1_filament_vessel_results.jpg', dpi=600)
	
	return [RMSE, RMSE_corr, f_opt, f_opt_corr]
	
###########################
def BmagnMultiModule(I_filaments,R_filaments,z_filaments,r_mirnv,z_mirnv,nfil):
	import numpy as np
	vectorR = np.zeros((12,nfil), np.float32)
	vectorZ = np.zeros((12,nfil), np.float32)
	unit_vecR = np.zeros((12,nfil), np.float32)
	unit_vecZ = np.zeros((12,nfil), np.float32)
	norm_vecR = np.zeros((12,nfil), np.float32)
	norm_vecZ = np.zeros((12,nfil), np.float32)
	
	for i in range(12):
			for j in range(0,nfil):
				vectorZ[i][j] = z_filaments[j]-z_mirnv[i]
				vectorR[i][j] = R_filaments[j]-r_mirnv[i]
	
	for i in range(12):
			for j in range(0,nfil):
				unit_vecZ[i][j] = np.divide(vectorZ[i][j], np.linalg.norm([vectorZ[i][j],vectorR[i][j]]) )
				unit_vecR[i][j] = np.divide(vectorR[i][j], np.linalg.norm([vectorZ[i][j],vectorR[i][j]]) )	
	
	for i in range(12):
			for j in range(0,nfil):
				norm_vecR[i][j] = -unit_vecZ[i][j]
				norm_vecZ[i][j] = unit_vecR[i][j]

	#I_filaments = [I_filament, I_filament2, I_filament3, I_filament4, I_filament5, I_filament6, I_filament7]
	Bz = np.zeros((12,nfil), np.float32)
	BR = np.zeros((12,nfil), np.float32)
	for i in range(12):
		data = BmagnmirnvMulti(z_filaments[0],R_filaments[0],I_filaments[0],r_mirnv[i],z_mirnv[i]) #return [ Bz, BR ]
		Bz[i][0] = data[0]
		BR[i][0] = data[1]
		for j in range(1,nfil):
			data = BmagnmirnvMulti(z_filaments[j],R_filaments[j],I_filaments[j],r_mirnv[i],z_mirnv[i])
			Bz[i][j] = data[0]
			BR[i][j] = data[1]	
	#Calculate the projection 
	Bmirn = np.zeros((12), np.float32)
	for i in range(12):
		for j in range(nfil):
			Bmirn[i] = Bmirn[i] + np.dot(Bz[i][j],norm_vecZ[i][j]) + np.dot(BR[i][j],norm_vecR[i][j])


	Bmirn = 0.01 * Bmirn

	return Bmirn

######################

def BmagnmirnvMulti( Z_filament,R_filament,I_filaments,r_mirnv,z_mirnv):
	import numpy as np
	from numpy import pi
	Zc = Z_filament
	I = I_filaments
	turns = 1															 # I just have one filament of one turn
	N = 100  								   							 # No of grids in the coil ( X-Y plane)
	u0 = 4*pi*0.001  													 # [microWb/(A cm)]
	phi = np.linspace(0, 2*pi, N) 									 	 # For describing a circle (coil)
	Rc = R_filament * np.cos(phi) 										 #R-coordinates of the filament
	Yc = R_filament * np.sin(phi) 										 #Y-coordinates of the filament
	
	#PYTHON RETURN ERROR IF I DON'T PREALLOCATE A VECTOR
	#Lets obtain the position vectors from dl to the mirnov 
	#mirnov is localized in the plane (y=0)

	RR = np.zeros(N)
	Rz = np.zeros(N)
	Ry = np.zeros(N)
	dlR = np.zeros(N)
	dly = np.zeros(N)
	for i in range(N-1):
		RR[i] = r_mirnv - 0.5* (Rc[i]+Rc[i+1]) 
		Rz[i] = z_mirnv - Zc
		Ry[i] = -0.5 * (Yc[i]+Yc[i+1])
		dlR[i]= Rc[i+1]-Rc[i]
		dly[i]=Yc[i+1]-Yc[i]
		
	RR[-1] = r_mirnv - 0.5*(Rc[-1]+Rc[0])
	Rz[-1] = z_mirnv - Zc
	Ry[-1] = -0.5 * (Rc[-1] + Rc[0])
	dlR[-1] = -Rc[-1] + Rc[0]
	dly[-1] = -Yc[-1]+Yc[0]
	
	#dl x r
	Rcross = np.multiply(-dly,Rz) #or -dly * Rz
	Ycross = np.multiply(dlR,Rz)
	Zcross = np.multiply(dly,RR) - np.multiply(dlR,Ry)
	R = np.sqrt(np.square(RR) + np.square(Rz) + np.square(Ry)) #OR 	#R = sqrt(RR**2 + Rz**2 + Ry**2) #R = np.sqrt(RR**2 + Rz**2 + Ry**2)
	
	#dB=m0/4pi (Idl x r)/r^2 
	BR1 = np.multiply(np.divide(I*u0, 4*pi*R**3), Rcross)
	Bz1 = np.multiply(np.divide(I*u0, 4*pi*R**3), Zcross)
	By1 = np.multiply(np.divide(I*u0, 4*pi*R**3), Ycross)
	
	#Initialize sum magnetic field to be zero first
	BR = 0
	Bz = 0
	By = 0
	BR = BR + np.sum(BR1)
	Bz = Bz + np.sum(Bz1)
	By = By + np.sum(By1)
	
	BR = BR * turns
	By = By * turns
	Bz = Bz * turns #units=[uWb / cm^2]	
	
	return  [Bz, BR]

######################

def SavePlot (x,y,name,unitX,unitY,SHOT_NUMBER):
	import numpy as np
	import string
	from numpy import pi
	import matplotlib.pyplot as plot
	plot.figure()
	plot.plot(x, y, label=name)
	plot.title("SHOT#"+str(SHOT_NUMBER)+" " +name)
	plot.xlabel(unitX)
	plot.ylabel(unitY)
	plot.grid()
	plot.legend(loc='upper right')
	plot.savefig(name+'.jpg', dpi=600)


#######################