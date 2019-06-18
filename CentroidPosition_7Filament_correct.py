"""
Created on Mon Feb  18 12:58:27 2019

@author: Il_Ciancio
"""

###########################
def BmagnMultiModule(Z_filament,R_filament,I_filaments,R_filaments,z_filaments,r_mirnv,z_mirnv):
	import numpy as np
	vectorR = np.zeros((12,7), np.float32)
	vectorZ = np.zeros((12,7), np.float32)
	unit_vecR = np.zeros((12,7), np.float32)
	unit_vecZ = np.zeros((12,7), np.float32)
	norm_vecR = np.zeros((12,7), np.float32)
	norm_vecZ = np.zeros((12,7), np.float32)
	
	for i in range(12):
			for j in range(0,7):
				vectorZ[i][j] = z_filaments[j]-z_mirnv[i]
				vectorR[i][j] = R_filaments[j]-r_mirnv[i]
	
	for i in range(12):
			for j in range(0,7):
				unit_vecZ[i][j] = np.divide(vectorZ[i][j], np.linalg.norm([vectorZ[i][j],vectorR[i][j]]) )
				unit_vecR[i][j] = np.divide(vectorR[i][j], np.linalg.norm([vectorZ[i][j],vectorR[i][j]]) )	
	
	for i in range(12):
			for j in range(0,7):
				norm_vecR[i][j] = -unit_vecZ[i][j]
				norm_vecZ[i][j] = unit_vecR[i][j]

	#I_filaments = [I_filament, I_filament2, I_filament3, I_filament4, I_filament5, I_filament6, I_filament7]
	Bz = np.zeros((12,7), np.float32)
	BR = np.zeros((12,7), np.float32)
	for i in range(12):
		data = BmagnmirnvMulti(Z_filament,R_filament,I_filaments[0],r_mirnv[i],z_mirnv[i]) #return [ Bz, BR ]
		Bz[i][0] = data[0]
		BR[i][0] = data[1]
		for j in range(1,7):
			data = BmagnmirnvMulti(z_filaments[j],R_filaments[j],I_filaments[j],r_mirnv[i],z_mirnv[i])
			Bz[i][j] = data[0]
			BR[i][j] = data[1]	
	#Calculate the projection 
	Bmirn = np.zeros((12), np.float32)
	for i in range(12):
		for j in range(7):
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

def ErrorMirnFuncMultiFilam(parameters, Mirnv_B_exp, R_filaments, z_filaments, r_mirnv, z_mirnv):
	import numpy as np
	
	Z_filament, R_filament, I_filament, I_filament2, I_filament3, I_filament4, I_filament5, I_filament6, I_filament7 = parameters
	
	vectorR = np.zeros((12,7), np.float32)
	vectorZ = np.zeros((12,7), np.float32)
	unit_vecR = np.zeros((12,7), np.float32)
	unit_vecZ = np.zeros((12,7), np.float32)
	norm_vecR = np.zeros((12,7), np.float32)
	norm_vecZ = np.zeros((12,7), np.float32)
	
	for i in range(12):
			for j in range(0,7):
				vectorZ[i][j] = z_filaments[j]-z_mirnv[i]
				vectorR[i][j] = R_filaments[j]-r_mirnv[i]
	
	for i in range(12):
			for j in range(0,7):
				unit_vecZ[i][j] = np.divide(vectorZ[i][j], np.linalg.norm([vectorZ[i][j],vectorR[i][j]]) )
				unit_vecR[i][j] = np.divide(vectorR[i][j], np.linalg.norm([vectorZ[i][j],vectorR[i][j]]) )	
	
	for i in range(12):
			for j in range(0,7):
				norm_vecR[i][j] = -unit_vecZ[i][j]
				norm_vecZ[i][j] = unit_vecR[i][j]
					
	
	I_filaments = [I_filament, I_filament2, I_filament3, I_filament4, I_filament5, I_filament6, I_filament7]
	
	Bz = np.zeros((12,7), np.float32)
	BR = np.zeros((12,7), np.float32)
	for i in range(12):
		data = BmagnmirnvMulti(Z_filament,R_filament,I_filaments[0],r_mirnv[i],z_mirnv[i]) #return [ Bz, BR ]
		Bz[i][0] = data[0]
		BR[i][0] = data[1]
		for j in range(1,7):
			data = BmagnmirnvMulti(z_filaments[j],R_filaments[j],I_filaments[j],r_mirnv[i],z_mirnv[i])
			Bz[i][j] = data[0]
			BR[i][j] = data[1]	

	Bmirn = np.zeros((12), np.float32)
	for i in range(12):
		for j in range(7):
			Bmirn[i] = Bmirn[i] + np.dot(Bz[i][j],norm_vecZ[i][j]) + np.dot(BR[i][j],norm_vecR[i][j])
	#Calculate the projection 

	Bmirn = 0.01 * Bmirn
	error = np.sum( np.abs(Mirnv_B_exp - Bmirn ) )
	return error
	
	
	
#####################


######################################################################
########## Plasma current centroid position reconstruction  ##########
########## Multifilaments,7 filaments, 9 freedom degrees    ##########
######################################################################

def centroidPosition_7_filaments(times, Mirnv_flux, Mirnv_flux_corr, SHOT_NUMBER, OPTIMIZATION, Ip):
	import numpy as np
	import string
	from numpy import pi
	from scipy.optimize import fmin_cobyla
	from scipy.optimize import fmin
	from scipy.optimize import fmin_slsqp
	import matplotlib.pyplot as plot
	from timeit import default_timer as timer
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
	radius = 4 # in [cm] (distance from the center of the chamber to the filaments)
	for i in range(1,7) :
		R_filaments = np.append( R_filaments,(46) + radius * np.cos( np.deg2rad(degr_filament) ))
		z_filaments = np.append( z_filaments, radius * np.sin( np.deg2rad(degr_filament) ) )
		degr_filament = degr_filament + 60;
	
	#EXPERIMENTAL MESUREMENTS [WB], I PRE MULTIPLIED Mirnv_10_fact=1.2823
	time_index = np.where(time == 120)
	print(time_index)
	#Find the exprimental values for that time moment
	
	Mirnov_flux = [i[time_index] for i in Mirnv_flux] #without external flux correction
	Mirnov_flux_corr = [i[time_index] for i in Mirnv_flux_corr] #with external flux correction
	
	#Let's go from [Wb] to [T]
	Mirnv_B_exp = np.divide(Mirnov_flux, ( 50 * 49e-6) ) # [T]
	Mirnv_B_exp_corr = np.divide(Mirnov_flux_corr, ( 50 * 49e-6) ) # [T]
	
	##### Optimization function, 7 filaments, 9 degrees of freedom
	##### Central filament - 3 dregrees of freedom (z,R,I)
	##### 6 sorrounding filaments - 1 degree of freedom (I)
	##########################################
	# ErrorMirnFuncMultiFilam(parameters, Mirnv_B_exp, R_filaments, z_filaments, R_mirn, z_mirn):
	# Z_filament, R_filament, I_filament, I_filament2, I_filament3, I_filament4, I_filament5,I_filament6, I_filament7= parameters
	##########################################
	def constr1(x, Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov):#low_bnd=[0,0,0,0,0,0,0,0,0]
		return [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]]
	def constr2(x, Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov):#high_bnd=[1,55,4000,4000,4000,4000,4000,4000,4000]
		return [1-x[0], 55-x[1], 4000-x[2], 4000-x[3], 4000-x[4], 4000-x[5], 4000-x[6], 4000-x[7], 4000-x[8]]
	def constr3(x, Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov):#high_bnd=[1,55,4000,4000,4000,4000,4000,4000,4000]
		return [1-x[0], 55-x[1], 4000-x[2], 4000-x[3], 4000-x[4], 4000-x[5], 4000-x[6], 4000-x[7], 4000-x[8]]
	f_opt_constr = np.array([], np.float32)
	f_opt_constr_corr = np.array([], np.float32)
	f_opt = np.array([], np.float32)
	f_opt_corr = np.array([], np.float32)
	mybounds = [(0,1),(0,55),(0,4000),(0,4000),(0,4000),(0,4000),(0,4000),(0,4000),(0,4000)]
	if OPTIMIZATION:
		print("Start optimization for the shot number: "+str(SHOT_NUMBER)+"!")
		start = timer()
		f_opt_constr = fmin_cobyla( ErrorMirnFuncMultiFilam,[0.5, 46.5, 500, 500, 500, 500, 500, 500, 500], [constr1, constr2], args=(Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov), consargs=None, rhobeg=0.1, rhoend=0.00000000000001, maxfun=100000, disp=None, catol=0.0000000002 )
		f_opt = fmin( ErrorMirnFuncMultiFilam, [0.5, 46.5, 500, 500, 500, 500, 500, 500, 500], args=(Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov) )
		end = timer()
		print ("Time for optimization: "+str(end-start))
		
		start = timer()
		f_opt_constr_corr = fmin_cobyla( ErrorMirnFuncMultiFilam,[0.5, 46.5, 500, 500, 500, 500, 500, 500, 500], [constr1, constr3], args=(Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov), consargs=None, rhobeg=0.1, rhoend=0.00000000000001, maxfun=100000, disp=None, catol=0.0000000002)
		f_opt_corr = fmin( ErrorMirnFuncMultiFilam, [0.5, 46.5, 500, 500, 500, 500, 500, 500, 500], args=(Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov))
		end = timer()
		print ("Time for corrected optimization: "+str(end-start))
		
		np.save('f_opt_constr_7Filaments_9degree_'+str(SHOT_NUMBER)+'.npy',f_opt_constr)
		np.save('f_opt_constr_corr_7Filaments_9degre_'+str(SHOT_NUMBER)+'.npy',f_opt_constr_corr)
		np.save('f_opt_7Filaments_9degre_'+str(SHOT_NUMBER)+'.npy',f_opt)
		np.save('f_opt_corr_7Filaments_9degre_'+str(SHOT_NUMBER)+'.npy',f_opt_corr)
	else:
		print("Loading data from shot number: "+str(SHOT_NUMBER)+"!")
		f_opt_constr = np.load('f_opt_constr_7Filaments_9degree_'+str(SHOT_NUMBER)+'.npy')
		f_opt_constr_corr = np.load('f_opt_constr_7Filaments_9degree_'+str(SHOT_NUMBER)+'.npy')
		f_opt = np.load('f_opt_7Filaments_9degre_'+str(SHOT_NUMBER)+'.npy',f_opt)
		f_opt_corr = np.load('f_opt_corr_7Filaments_9degre_'+str(SHOT_NUMBER)+'.npy',f_opt_corr)
		
	
	#Lets check how close is our minimization values to the experimental ones by applaying Biot-Savart with them 
	#constrained
	Mirnov_flux_experimental_multi_constr = np.array([], np.float32)
	Mirnov_flux_corr_experimental_multi_constr = np.array([], np.float32)
	

	Mirnov_flux_experimental_multi_constr = np.append( Mirnov_flux_experimental_multi_constr, BmagnMultiModule(f_opt_constr[0], f_opt_constr[1], f_opt_constr[2:9], R_filaments, z_filaments, R_mirnov, z_mirnov ) )
	Mirnov_flux_corr_experimental_multi_constr = np.append( Mirnov_flux_corr_experimental_multi_constr, BmagnMultiModule(f_opt_constr_corr[0], f_opt_constr_corr[1], f_opt_constr_corr[2:9], R_filaments, z_filaments, R_mirnov, z_mirnov) )
	#unconstrained
	Mirnov_flux_experimental_multi = np.array([], np.float32)
	Mirnov_flux_corr_experimental_multi = np.array([], np.float32)
	

	Mirnov_flux_experimental_multi = np.append( Mirnov_flux_experimental_multi, BmagnMultiModule(f_opt[0], f_opt[1], f_opt[2:9], R_filaments, z_filaments, R_mirnov, z_mirnov  ) )
	Mirnov_flux_corr_experimental_multi = np.append( Mirnov_flux_corr_experimental_multi, BmagnMultiModule(f_opt_corr[0], f_opt_corr[1], f_opt_corr[2:9], R_filaments, z_filaments, R_mirnov, z_mirnov ) )

	#compute de error
	RMSE_const = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_experimental_multi_constr, Mirnv_B_exp ) ) ) )
	RMSE_const_corr = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_corr_experimental_multi_constr, Mirnv_B_exp_corr ) ) ) )
	RMSE = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_experimental_multi, Mirnv_B_exp ) ) ) )
	RMSE_corr = np.sqrt( np.square( np.mean( np.subtract(Mirnov_flux_corr_experimental_multi, Mirnv_B_exp_corr ) ) ) )
	
		
	#PLOTTIAMO
	plot.figure(1)
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA]")
	plot.plot(time, Ip, label='Plasma Current')
	plot.plot(time[time_index], Ip[time_index], '.k', MarkerSize=10, label='Chosen Flat-Top')
	plot.legend(loc='upper right')
	plot.xlabel('time[ms]')
	plot.ylabel('I[A]')
	plot.savefig('7_filament_flattopchosen.jpg', dpi=600)
	
	plot.figure(2)
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp, '-o',label="Experimental Data")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_experimental_multi_constr, '-*',label="Biot-savart(constrained optimization)")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (Multifilament without external flux correction)")
	plot.legend([p1, p2], loc='upper right')
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	plot.savefig('7_filament_RMSE_constrained.jpg', dpi=600)
	
	plot.figure(3)
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp_corr, '-o',label="Experimental Data corrected (Multifilament)")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_experimental_multi_constr, '-*',label="Biot-savart(constrained optimization)")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (Multifilament with external flux correction)")
	plot.legend([p1, p2])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	plot.legend(loc='upper right')
	plot.savefig('7_filament_RMSE_corr_constrained.jpg', dpi=600)
	#Plasma, vessel and mirnov coil plot
	plot.figure(4)
	plot.title("Geometry inside the vessel, contrained optimization")
	plot.plot(x_vessel,y_vessel,color='k', linewidth=2.0)
	plot.plot(46,0,'.m',MarkerSize = 480)
	plot.plot(f_opt_constr[1],f_opt_constr[0],'.k',MarkerSize = 20, label="without external flux  correction")
	plot.plot(f_opt_constr_corr[1],f_opt_constr_corr[0],'.r',MarkerSize = 10, label="with external flux correction")
	plot.text(f_opt_constr[1]-1,f_opt_constr[0]+0.5, str(int(f_opt_constr[2]))+"[A]")
	plot.text(f_opt_constr_corr[1]-1,f_opt_constr_corr[0]+1, str(int(f_opt_constr_corr[2]))+"[A]", color='red')

	for i in range(1,7):
		plot.plot(R_filaments[i],z_filaments[i],'.b',MarkerSize = 20)
		plot.text(R_filaments[i]-1,z_filaments[i]+0.5, str(int(f_opt_constr[2+i]))+"[A]")
		plot.text(R_filaments[i]-1,z_filaments[i]+1, str(int(f_opt_constr_corr[2+i]))+"[A]", color='red')
	for i in range(12):
		plot.text(R_mirnov[i], z_mirnov[i], str(i+1), horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	
	plot.text(57,0,'LFS',FontSize = 15)
	plot.text(33,0,'HFS',FontSize = 15)
	plot.xlabel('R[cm]')
	plot.ylabel('Z[cm]')
	plot.grid()
	plot.axis('equal')
	plot.legend(loc='upper right')
	plot.savefig('7_filament_vessel_results_constrained.jpg', dpi=600)
	
	plot.figure(5)
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp, '-o',label="Experimental Data")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_experimental_multi, '-*',label="Biot-savart(uncontrained optimization )")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (Multifilament)")
	plot.legend([p1, p2])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	plot.savefig('7_filament_RMSE_unconstrained.jpg', dpi=600)
	
	plot.figure(6)
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp_corr, '-o',label="Experimental Data corrected (Multifilament)")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_experimental_multi, '-*',label="Biot-savart(uncontrained optimization)")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=" + str(time[time_index])+ "[ms]  Ip~4.1[kA] (External flux corrected)")
	plot.legend([p1, p2])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	plot.savefig('7_filament_RMSE_corr_unconstrained.jpg', dpi=600)
	#Plasma, vessel and mirnov coil plot
	plot.figure(7)
	plot.title("Geometry inside the vessel, unconstrained optimization")
	plot.plot(x_vessel,y_vessel,color='k', linewidth=2.0)
	plot.plot(46,0,'.m',MarkerSize = 480)
	plot.plot(f_opt[1],f_opt[0],'.k',MarkerSize = 20, label="without external flux correction")
	plot.plot(f_opt_corr[1],f_opt_corr[0],'.r',MarkerSize = 10, label="with external flux correction")
	plot.text(f_opt[1],f_opt_constr[0]+1, str(int(f_opt_constr[2]))+"[A]")
	plot.text(f_opt_corr[1],f_opt_constr_corr[0]+1.5, str(int(f_opt_constr_corr[2]))+"[A]", color='red')

	for i in range(1,7):
		plot.plot(R_filaments[i],z_filaments[i],'.b',MarkerSize = 20)
		plot.text(R_filaments[i]-1,z_filaments[i]-0.5, str(int(f_opt[2+i]))+"[A]")
		plot.text(R_filaments[i]-1,z_filaments[i]-1, str(int(f_opt_corr[2+i]))+"[A]", color='red')
	for i in range(12):
		plot.text(R_mirnov[i], z_mirnov[i], str(i+1), horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	
	plot.text(57,0,'LFS',FontSize = 15)
	plot.text(33,0,'HFS',FontSize = 15)
	plot.xlabel('R[cm]')
	plot.ylabel('Z[cm]')
	plot.grid()
	plot.axis('equal')
	plot.legend(loc='upper right')
	plot.savefig('7_filament_vessel_results_unconstrained.jpg', dpi=600)
	#Plasma, vessel and mirnov coil plot
	plot.figure(8)
	plot.title("Geometry inside the vessel")
	plot.plot(x_vessel,y_vessel,color='k', linewidth=2.0)
	plot.plot(46,0,'.k',MarkerSize = 20)
	plot.text(44,0, "If_1")
	plot.arrow(46,0, 5,0, head_width=0.5, head_length=0.5)
	plot.text(48,0, "r")
	plot.arrow(46,0, 0,5, head_width=0.5, head_length=0.5)
	plot.text(46,2.5, "z")

	for i in range(1,7):
		plot.plot(R_filaments[i],z_filaments[i],'.b',MarkerSize = 20)
		plot.text(R_filaments[i]-0.5,z_filaments[i]+0.5, "If_"+str(i+1))
	for i in range(12):
		plot.text(R_mirnov[i], z_mirnov[i], str(i+1), horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	
	plot.text(57,0,'LFS',FontSize = 15)
	plot.text(33,0,'HFS',FontSize = 15)
	plot.xlabel('R[cm]')
	plot.ylabel('Z[cm]')
	plot.grid()
	plot.axis('equal')
	plot.savefig('7_filament_vessel_geometry.jpg', dpi=600)
	return [f_opt, f_opt_corr, f_opt_constr, f_opt_constr_corr, Mirnov_flux_experimental_multi_constr, Mirnov_flux_corr_experimental_multi_constr, RMSE_const_corr, RMSE_const]