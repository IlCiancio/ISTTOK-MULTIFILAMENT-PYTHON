"""
Created on Mon Feb  18 12:58:27 2019

@author: Il_Ciancio
"""

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

def ErrorMirnFuncMultiFilam(parameters, Mirnv_B_exp, R_filaments, z_filaments, r_mirnv, z_mirnv, nfil):
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
					
	
	I_filaments = parameters
	
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

	Bmirn = np.zeros((12), np.float32)
	for i in range(12):
		for j in range(nfil):
			Bmirn[i] = Bmirn[i] + np.dot(Bz[i][j],norm_vecZ[i][j]) + np.dot(BR[i][j],norm_vecR[i][j])
	#Calculate the projection 

	Bmirn = 0.01 * Bmirn
	error = np.sum( np.abs(Mirnv_B_exp - Bmirn ) )
	return error
	
	
	
#####################


######################################################################
########## Plasma current centroid position reconstruction  ##########
########## Multifilaments,N filaments, N freedom degrees    ##########
######################################################################

def centroidPosition_N_filaments(nfilments, times, Mirnv_flux, Mirnv_flux_corr, SHOT_NUMBER, OPTIMIZATION, Ip_magn_corr_value, sumIfil_value):
	import numpy as np
	import string
	from numpy import pi
	import scipy as scp
	from scipy.optimize import fmin
	import matplotlib.pyplot as plot
	from timeit import default_timer as timer
	time = 1e-03 * times #TIME IN ms
	N_FILAMENTS = nfilments
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
	degr_filament = 0
	deg_fact = 360 / (N_FILAMENTS)
	radius = 5.5 # in [cm] (distance from the center of the chamber to the filaments)
	for i in range(0, N_FILAMENTS) :
		R_filaments = np.append( R_filaments,(46) + radius * np.cos( np.deg2rad(degr_filament) ))
		z_filaments = np.append( z_filaments, radius * np.sin( np.deg2rad(degr_filament) ) )
		degr_filament = degr_filament + deg_fact;
	
	#EXPERIMENTAL MESUREMENTS [WB], I PRE MULTIPLIED Mirnv_10_fact=1.2823
	time_index = np.where(time == 390)
	print(Ip_magn_corr_value[time_index])
	#Find the exprimental values for that time moment
	Mirnov_flux = [i[time_index] for i in Mirnv_flux] #without external flux correction
	Mirnov_flux_corr = [i[time_index] for i in Mirnv_flux_corr] #with external flux correction
	
	#Let's go from [Wb] to [T]
	Mirnv_B_exp = np.divide(Mirnov_flux, ( 50 * 49e-6) ) # [T]
	Mirnv_B_exp_corr = np.divide(Mirnov_flux_corr, ( 50 * 49e-6) ) # [T]
	
	##### Optimization function, N filaments, N degrees of freedom
	##### N sorrounding filaments - 1 degree of freedom (I)
	##########################################
	# ErrorMirnFuncMultiFilam(parameters, Mirnv_B_exp, R_filaments, z_filaments, R_mirn, z_mirn):
	# Z_filament, R_filament, I_filament, I_filament2, I_filament3, I_filament4, I_filament5,I_filament6, I_filament7= parameters
	##########################################

	f_opt = np.array([], np.float32)
	f_opt_corr = np.array([], np.float32)
	x0 = np.array([], np.float32)
	for i in range(0, N_FILAMENTS):
		x0 = np.append(x0, 500)
	
	if OPTIMIZATION:
		print("Start optimization for the shot number: "+str(SHOT_NUMBER)+"!")
		start = timer()
		f_opt = fmin( ErrorMirnFuncMultiFilam, x0, args=(Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS) )
		end = timer()
		print ("Time for optimization: "+str(end-start))
		
		start = timer()
		f_opt_corr = fmin( ErrorMirnFuncMultiFilam, x0, args=(Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS))
		end = timer()
		print ("Time for corrected optimization: "+str(end-start))
		
		np.save('f_opt_NFilaments_Ndegree_'+str(SHOT_NUMBER)+'.npy',f_opt)
		np.save('f_opt_corr_NFilaments_Ndegree_'+str(SHOT_NUMBER)+'.npy',f_opt_corr)
	else:
		print("Loading data from shot number: "+str(SHOT_NUMBER)+"!")
		f_opt = np.load('f_opt_NFilaments_Ndegree_'+str(SHOT_NUMBER)+'.npy')
		f_opt_corr = np.load('f_opt_corr_NFilaments_Ndegree_'+str(SHOT_NUMBER)+'.npy')
	
	#Lets check how close is our minimization values to the experimental ones by applaying Biot-Savart with them 
	Mirnov_flux_experimental_multi = np.array([], np.float32)
	Mirnov_flux_corr_experimental_multi = np.array([], np.float32)
	

	Mirnov_flux_experimental_multi = np.append( Mirnov_flux_experimental_multi, BmagnMultiModule(f_opt, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS ) )
	Mirnov_flux_corr_experimental_multi = np.append( Mirnov_flux_corr_experimental_multi, BmagnMultiModule(f_opt_corr, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS) )

	#compute de error
	RMSE = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_experimental_multi, Mirnv_B_exp ) ) ) )
	RMSE_corr = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_corr_experimental_multi, Mirnv_B_exp_corr ) ) ) )
	np.save('RMSE_NFilaments_Ndegree_'+str(SHOT_NUMBER)+'.npy',RMSE)
	np.save('RMSE_corr_NFilaments_Ndegree_'+str(SHOT_NUMBER)+'.npy',RMSE_corr)
	# Matrix whose elements gives the contribution  to the measuremnt i  to a unitary current in the filament j [T]
	Mfp = np.zeros((12,N_FILAMENTS), np.float32)
	from function import Bmagnmirnv #def Bmagnmirnv( Z_filament,R_filament,I_filament,r_mirnv,z_mirnv):
	for i in range(12):
		for j in range(N_FILAMENTS):
			Mfp[i][j] = Bmagnmirnv(z_filaments[j], R_filaments[j], 1, R_mirnov[i], z_mirnov[i])
	start = timer()
	Mpf = np.zeros((12,N_FILAMENTS), np.float32)
	#Mpf = np.linalg.pinv(Mfp, rcond = 0)
	Mpf = scp.linalg.pinv(Mfp)
	end = timer()
	print ("Time for pinv: "+str(end-start))
	I_Mpf_corr = np.dot(Mpf,Mirnv_B_exp_corr)
	np.save('I_Mpf_corr_NFilaments_Ndegree_'+str(SHOT_NUMBER)+'.npy',I_Mpf_corr)
	Mirnov_flux_corr_theo_multi = np.array([], np.float32)
	Mirnov_flux_corr_theo_multi = np.append( Mirnov_flux_corr_theo_multi, BmagnMultiModule(I_Mpf_corr, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS) )
	RMSE_theo_corr = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_corr_theo_multi, Mirnv_B_exp_corr ) ) ) )
	np.save('RMSE_theo_corr_NFilaments_Ndegree_'+str(SHOT_NUMBER)+'.npy',RMSE_theo_corr)

	Tesla = np.divide((Mirnv_flux_corr),(49*50*1e-6))
	I_filament_all = np.dot(Mpf,Tesla)
	sum_Ifil = np.sum(I_filament_all,0)
	#PLOTTIAMO
	
	plot.figure()
	plot.plot(time, Ip_magn_corr_value, label='Measured Plasma Current')
	plot.plot(time, sum_Ifil, label='Plasma Current Reconstructed in Pyhton')
	plot.plot(time, sumIfil_value, label='Plasma Current Reconstructed in RealTime')
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA]")
	plot.xlabel('time[ms]')
	plot.ylabel('I[A]')
	plot.grid()
	plot.legend(loc='upper right')
	plot.savefig('compare_currents.jpg', dpi=600)
	
	plot.figure()
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp, '-o',label="Experimental Data")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_experimental_multi, '-*',label="Biot-savart(optimized )")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (Multifilament)")
	plot.legend([p1, p2])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	
	plot.figure()
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp_corr, '-o',label="Experimental Data corrected (Multifilament)")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_experimental_multi, '-*',label="Biot-savart(optimized )")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (External flux corrected)")
	plot.legend([p1, p2])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	
	plot.figure()
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp_corr, '-o',label="Experimental Data corrected (Multifilament)")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_theo_multi, '-*',label="Pseudo inverse method")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (External flux corrected)")
	plot.legend([p1, p2])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	
	plot.figure()
	plot.plot(time, Ip_magn_corr_value, label='Plasma Current')
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (Multifilament)")
	plot.xlabel('time[ms]')
	plot.ylabel('I[A]')
	plot.grid()
	plot.legend(loc='upper right')
	plot.text(20, 4500, "A", horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	plot.text(100, -4460, "B", horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	plot.text(32, 0, "C", horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	plot.savefig('plasma_current.jpg', dpi=600)
	
	
	#Plasma, vessel and mirnov coil plot
	plot.figure()
	plot.plot(x_vessel,y_vessel,color='k', linewidth=2.0)
	#plot.plot(46,0,'.m',MarkerSize = 620)
	for i in range(0,N_FILAMENTS):
		plot.plot(R_filaments[i],z_filaments[i],'.b',MarkerSize = 20)
	for i in range(12):
		plot.text(R_mirnov[i], z_mirnov[i], str(i+1), horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	
	plot.text(57,0,'LFS',FontSize = 15)
	plot.text(33,0,'HFS',FontSize = 15)
	plot.arrow(46,0,4.5,0,head_width=0.5, head_length=0.5, fc='k', ec='k')
	plot.text(47.5,-1,'r=5.5[cm]',FontSize = 10)
	plot.xlabel('R[cm]')
	plot.ylabel('Z[cm]')
	plot.grid()
	plot.title("Geometry selected for ISTTok")
	plot.axis('equal')
	
	plot.savefig('geometry_inISTtok.jpg', dpi=600)

	# plot.figure()
	# plot.plot(x_vessel,y_vessel,color='k', linewidth=2.0)
	# plot.plot(46,0,'.b',MarkerSize = 20)
	# plot.arrow(46,0,0,5,head_width=0.5, head_length=0.5, fc='k', ec='k')
	# plot.arrow(46,0,5,0,head_width=0.5, head_length=0.5, fc='k', ec='k')
	# plot.text(45,-1.25,'If',FontSize = 15)
	# plot.text(45,2,'z',FontSize = 15)
	# plot.text(48.5,-1,'r',FontSize = 15)
	# for i in range(12):
		# plot.text(R_mirnov[i], z_mirnov[i], str(i+1), horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	
	# plot.text(57,0,'LFS',FontSize = 15)
	# plot.text(33,0,'HFS',FontSize = 15)
	# plot.xlabel('R[cm]')
	# plot.ylabel('Z[cm]')
	# plot.grid()
	# plot.axis('equal')
	# plot.savefig('1_filament.jpg', dpi=600)
	
	

	return [Mfp, Mpf, I_Mpf_corr]
	
	
##########################

def centroidPosition_N_filaments_change_radius(raggio,nfilments, times, Mirnv_flux, Mirnv_flux_corr, SHOT_NUMBER, OPTIMIZATION, Ip_magn_corr_value, sumIfil_value):
	import numpy as np
	import string
	from numpy import pi
	import scipy as scp
	from scipy.optimize import fmin
	import matplotlib.pyplot as plot
	from timeit import default_timer as timer
	time = 1e-03 * times #TIME IN ms
	N_FILAMENTS = nfilments
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
	R_filaments = np.zeros(N_FILAMENTS)
	z_filaments = np.zeros(N_FILAMENTS)
	degr_filament = 0
	deg_fact = 360 / (N_FILAMENTS)
	for i in range(0,4):
		radius = raggio + i # in [cm] (distance from the center of the chamber to the filaments)
		for i in range(0, N_FILAMENTS) :
			R_filaments[i] = (46) + radius * np.cos( np.deg2rad(degr_filament) )
			z_filaments[i] = radius * np.sin( np.deg2rad(degr_filament) )
			degr_filament = degr_filament + deg_fact;
		
		#EXPERIMENTAL MESUREMENTS [WB], I PRE MULTIPLIED Mirnv_10_fact=1.2823
		time_index = np.where(time == 390)
		print(Ip_magn_corr_value[time_index])
		#Find the exprimental values for that time moment
		Mirnov_flux = [i[time_index] for i in Mirnv_flux] #without external flux correction
		Mirnov_flux_corr = [i[time_index] for i in Mirnv_flux_corr] #with external flux correction
		
		#Let's go from [Wb] to [T]
		Mirnv_B_exp = np.divide(Mirnov_flux, ( 50 * 49e-6) ) # [T]
		Mirnv_B_exp_corr = np.divide(Mirnov_flux_corr, ( 50 * 49e-6) ) # [T]
		
		##### Optimization function, N filaments, N degrees of freedom
		##### N sorrounding filaments - 1 degree of freedom (I)
		##########################################
		# ErrorMirnFuncMultiFilam(parameters, Mirnv_B_exp, R_filaments, z_filaments, R_mirn, z_mirn):
		# Z_filament, R_filament, I_filament, I_filament2, I_filament3, I_filament4, I_filament5,I_filament6, I_filament7= parameters
		##########################################

		f_opt = np.zeros(N_FILAMENTS)
		f_opt_corr = np.zeros(N_FILAMENTS)
		x0 = np.zeros(N_FILAMENTS)
		for i in range(0, N_FILAMENTS):
			x0[i] = 500
		
		if OPTIMIZATION:
			print("Start optimization for the shot number: "+str(SHOT_NUMBER)+"!")
			start = timer()
			f_opt = fmin( ErrorMirnFuncMultiFilam, x0, args=(Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS) )
			end = timer()
			print ("Time for optimization: "+str(end-start))
			
			start = timer()
			f_opt_corr = fmin( ErrorMirnFuncMultiFilam, x0, args=(Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS))
			end = timer()
			print ("Time for corrected optimization: "+str(end-start))
			
			np.save('f_opt_r_'+str(radius)+'_N'+str(N_FILAMENTS)+'_'+str(SHOT_NUMBER)+'.npy',f_opt)
			np.save('f_opt_r_'+str(radius)+'_N'+str(N_FILAMENTS)+str(SHOT_NUMBER)+'.npy',f_opt_corr)
		else:
			print("Loading data from shot number: "+str(SHOT_NUMBER)+"!")
			f_opt = np.load('f_opt_r_'+str(radius)+'_N'+str(N_FILAMENTS)+'_'+str(SHOT_NUMBER)+'.npy')
			f_opt_corr = np.load('f_opt_r_'+str(radius)+'_N'+str(N_FILAMENTS)+str(SHOT_NUMBER)+'.npy')
		
		#Lets check how close is our minimization values to the experimental ones by applaying Biot-Savart with them 
		Mirnov_flux_experimental_multi = np.zeros(12)
		Mirnov_flux_corr_experimental_multi = np.zeros(12)
		
		Mirnov_flux_experimental_multi = BmagnMultiModule(f_opt, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS ) 
		Mirnov_flux_corr_experimental_multi = BmagnMultiModule(f_opt_corr, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS) 

		#compute de error
		RMSE = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_experimental_multi, Mirnv_B_exp ) ) ) )
		RMSE_corr = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_corr_experimental_multi, Mirnv_B_exp_corr ) ) ) )
		np.save('RMSE_NFilaments_Ndegree_r_'+str(radius)+'_N'+str(N_FILAMENTS)+'_'+str(SHOT_NUMBER)+'.npy',RMSE)
		np.save('RMSE_corr_NFilaments_Ndegree_r_'+str(radius)+'_N'+str(N_FILAMENTS)+'_'+str(SHOT_NUMBER)+'.npy',RMSE_corr)
		# Matrix whose elements gives the contribution  to the measuremnt i  to a unitary current in the filament j [T]
		Mfp = np.zeros((12,N_FILAMENTS), np.float32)
		from function import Bmagnmirnv #def Bmagnmirnv( Z_filament,R_filament,I_filament,r_mirnv,z_mirnv):
		for i in range(12):
			for j in range(N_FILAMENTS):
				Mfp[i][j] = Bmagnmirnv(z_filaments[j], R_filaments[j], 1, R_mirnov[i], z_mirnov[i])
		start = timer()
		Mpf = np.zeros((12,N_FILAMENTS), np.float32)
		#Mpf = np.linalg.pinv(Mfp, rcond = 0)
		Mpf = scp.linalg.pinv(Mfp)
		end = timer()
		print ("Time for pinv: "+str(end-start))
		I_Mpf_corr = np.dot(Mpf,Mirnv_B_exp_corr)
		np.save('I_Mpf_corr_NFilaments_Ndegree_r_'+str(radius)+'_N'+str(N_FILAMENTS)+'_'+str(SHOT_NUMBER)+'.npy',I_Mpf_corr)
		Mirnov_flux_corr_theo_multi = np.zeros(12)
		Mirnov_flux_corr_theo_multi = BmagnMultiModule(I_Mpf_corr, R_filaments, z_filaments, R_mirnov, z_mirnov, N_FILAMENTS) 
		RMSE_theo_corr = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_corr_theo_multi, Mirnv_B_exp_corr ) ) ) )
		np.save('RMSE_theo_corr_NFilaments_Ndegree_r_'+str(radius)+'_N'+str(N_FILAMENTS)+'_'+str(SHOT_NUMBER)+'.npy',RMSE_theo_corr)

		Tesla = np.divide((Mirnv_flux_corr),(49*50*1e-6))
		I_filament_all = np.dot(Mpf,Tesla)
		sum_Ifil = np.sum(I_filament_all,0)
		#PLOTTIAMO
		
		plot.figure()
		plot.plot(time, Ip_magn_corr_value, label='Measured Plasma Current')
		plot.plot(time, sum_Ifil, label='Plasma Current Reconstructed in Pyhton')
		plot.plot(time, sumIfil_value, label='Plasma Current Reconstructed in RealTime')
		plot.title("#"+str(SHOT_NUMBER)+"t=" + str(time[time_index])+ "[ms]Ip~4.1[kA] r=" + str(radius)+"[cm],N="+str(N_FILAMENTS))
		plot.xlabel('time[ms]')
		plot.ylabel('I[A]')
		plot.grid()
		plot.legend(loc='upper right')
		plot.savefig('compare_currents'+str(radius)+'_'+str(N_FILAMENTS)+'.jpg', dpi=600)
		
		plot.figure()
		p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp, '-o',label="Experimental Data")
		p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_experimental_multi, '-*',label="Biot-savart(optimized)")
		plot.grid()
		plot.title("#"+str(SHOT_NUMBER)+"t=" + str(time[time_index])+ "[ms]Ip~4.1[kA](Multifilament)r=" + str(radius)+"[cm],N= "+str(N_FILAMENTS))
		plot.legend([p1, p2])
		plot.xlabel("Mirnov #")
		plot.ylabel("Optimization [mT]")
		plot.axis('equal')
		plot.savefig('RMSE_'+str(radius)+'_'+str(N_FILAMENTS)+'.jpg', dpi=600)
		
		plot.figure()
		p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp_corr, '-o',label="Experimental Data corrected (Multifilament)")
		p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_experimental_multi, '-*',label="Biot-savart(optimized )")
		plot.grid()
		plot.title("#"+str(SHOT_NUMBER)+"t=" + str(time[time_index])+ "[ms]Ip~4.1[kA](External flux corrected)r=" + str(radius)+"[cm],N= "+str(N_FILAMENTS))
		plot.legend([p1, p2])
		plot.xlabel("Mirnov #")
		plot.ylabel("Optimization [mT]")
		plot.axis('equal')
		plot.savefig('RMSE_corr_'+str(radius)+'_'+str(N_FILAMENTS)+'.jpg', dpi=600)
		plot.figure()
		p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp_corr, '-o',label="Experimental Data corrected (Multifilament)")
		p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_theo_multi, '-*',label="Pseudo inverse method")
		plot.grid()
		plot.title("#"+str(SHOT_NUMBER)+"t=" + str(time[time_index])+ "[ms]Ip~4.1[kA](External flux corrected)r= " + str(radius)+"[cm],N= "+str(N_FILAMENTS))
		plot.legend([p1, p2])
		plot.xlabel("Mirnov #")
		plot.ylabel("Optimization [mT]")
		plot.axis('equal')
		plot.savefig('RMSE_SVD_'+str(radius)+'_'+str(N_FILAMENTS)+'.jpg', dpi=600)
		
		#Plasma, vessel and mirnov coil plot
		plot.figure()
		plot.plot(x_vessel,y_vessel,color='k', linewidth=2.0)
		plot.plot(46,0,'.m',MarkerSize = 620)
		for i in range(0,N_FILAMENTS):
			plot.plot(R_filaments[i],z_filaments[i],'.b',MarkerSize = 20)
		for i in range(12):
			plot.text(R_mirnov[i], z_mirnov[i], str(i+1), horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
		
		plot.text(57,0,'LFS',FontSize = 15)
		plot.text(33,0,'HFS',FontSize = 15)
		plot.xlabel('R[cm]')
		plot.ylabel('Z[cm]')
		plot.grid()
		plot.axis('equal')
		plot.title("Geometry inside the vessel, r= " + str(radius)+" [cm], N_FILAMENTS = "+str(N_FILAMENTS))
		plot.savefig('geometry_'+str(radius)+'_'+str(N_FILAMENTS)+'.jpg', dpi=600)


	
	

	return [Mirnov_flux_corr_theo_multi,Mirnv_B_exp_corr, Mfp, Mpf, I_Mpf_corr]