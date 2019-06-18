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
		Bz[i][0], BR[i][0]  = BmagnmirnvMulti(z_filaments[0],R_filaments[0],I_filaments[0],r_mirnv[i],z_mirnv[i]) #return [ Bz, BR ]
		for j in range(1,7):
			Bz[i][j], BR[i][j] = BmagnmirnvMulti(z_filaments[j],R_filaments[j],I_filaments[j],r_mirnv[i],z_mirnv[i])

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
	BR = np.sum(BR1)
	Bz = np.sum(Bz1)
	By = np.sum(By1)
	
	BR = BR * turns
	By = By * turns
	Bz = Bz * turns #units=[uWb / cm^2]	
	
	return  [Bz, BR]

######################
def ErrorMirnFuncMultiFilam(parameters, Mirnv_B_exp, R_filaments, z_filaments, r_mirnv, z_mirnv):
	import numpy as np
	I_filament, I_filament2, I_filament3, I_filament4, I_filament5, I_filament6, I_filament7 = parameters
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
		Bz[i][0], BR[i][0]  = BmagnmirnvMulti(z_filaments[0],R_filaments[0],I_filaments[0],r_mirnv[i],z_mirnv[i]) #return [ Bz, BR ]
		for j in range(1,7):
			Bz[i][j], BR[i][j] = BmagnmirnvMulti(z_filaments[j],R_filaments[j],I_filaments[j],r_mirnv[i],z_mirnv[i])	

	Bmirn = np.zeros((12), np.float32)
	for i in range(12):
		for j in range(7):
			Bmirn[i] = Bmirn[i] + np.dot(Bz[i][j],norm_vecZ[i][j]) + np.dot(BR[i][j],norm_vecR[i][j])
	
	Bmirn = 0.01 * Bmirn
	error = np.sum( np.abs(Mirnv_B_exp - Bmirn ) )
	return error
	
	
	
#####################

######################################################################
######################################################################
########## Plasma current centroid position reconstruction  ##########
########## Multifilaments,7 filaments, 7 freedom degrees    ##########
######################################################################
######################################################################

def centroidPosition_7_filaments_7degree(times, Mirnv_flux, Mirnv_flux_corr, SHOT_NUMBER, OPTIMIZATION):
	import numpy as np
	import string
	from numpy import pi
	from scipy.optimize import fmin_cobyla
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
	radius = 5 # in [cm] (distance from the center of the chamber to the filaments)
	for i in range(1,7) :
		R_filaments = np.append( R_filaments,(46) + radius * np.cos( np.deg2rad(degr_filament) ))
		z_filaments = np.append( z_filaments, radius * np.sin( np.deg2rad(degr_filament) ) )
		degr_filament = degr_filament + 60;
	
	#EXPERIMENTAL MESUREMENTS [WB], I PRE MULTIPLIED Mirnv_10_fact=1.2823
	time_index = np.where(time == 175) 
	print(time_index)
	#Find the exprimental values for that time moment
	
	Mirnov_flux = [i[time_index] for i in Mirnv_flux] #without external flux correction
	Mirnov_flux_corr = [i[time_index] for i in Mirnv_flux_corr] #with external flux correction
	
	#Let's go from [Wb] to [T]
	Mirnv_B_exp = np.divide(Mirnov_flux,(50*49e-6)) # [T]
	Mirnv_B_exp_corr = np.divide(Mirnov_flux_corr,(50*49e-6)) # [T]
	
	##### Optimization function, 7 filaments, 9 degrees of freedom
	##### Central filament - 3 dregrees of freedom (z,R,I)
	##### 6 sorrounding filaments - 1 degree of freedom (I)
	##########################################
	# ErrorMirnFuncMultiFilam(parameters, Mirnv_B_exp, R_filaments, z_filaments, R_mirn, z_mirn):
	# Z_filament, R_filament, I_filament, I_filament2, I_filament3, I_filament4, I_filament5,I_filament6, I_filament7= parameters
	##########################################
	def constr1(x, Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov):#low_bnd=[0,0,0,0,0,0,0,0,0]
		return x[0], x[1], x[2], x[3], x[4], x[5], x[6]
		#return x
	def constr2(x, Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov):#high_bnd=[1,55,4000,4000,4000,4000,4000,4000,4000]
		return 4000-x[0], 4000-x[1], 4000-x[2], 4000-x[3], 4000-x[4], 4000-x[5], 4000-x[6]
	def constr3(x, Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov):#high_bnd=[1,55,4000,4000,4000,4000,4000,4000,4000]
		return 4000-x[0], 4000-x[1], 4000-x[2], 4000-x[3], 4000-x[4], 4000-x[5], 4000-x[6]
	def constr4(x, Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov):#low_bnd=[0,0,0,0,0,0,0,0,0]
		return x[0], x[1], x[2], x[3], x[4], x[5], x[6]
	def constr5(x, Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov):
		return 4000 - x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]
	def constr6(x, Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov):
		return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6]  - 2000
	f_opt = np.array([], np.float32)
	f_opt_corr = np.array([], np.float32)
	mybounds = [(0,4000),(0,4000),(0,4000),(0,4000),(0,4000),(0,4000),(0,4000)]
	if OPTIMIZATION:
		print("Start optimization for the shot number: "+str(SHOT_NUMBER)+"!")
		

		
		start = timer()
		f_opt = fmin_cobyla( ErrorMirnFuncMultiFilam,[1000, 500, 500, 500, 500, 500, 500],  [constr1] + [constr2], args=(Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov), consargs=None, rhobeg=0.1, rhoend=1e-10, maxfun=20000, catol=1e-10)

		end = timer()
		print ("Time for optimization: "+str(end-start))
		start = timer()
		f_opt_corr = fmin_cobyla( ErrorMirnFuncMultiFilam,[1000, 500, 500, 500, 500, 500, 500], [constr4] + [constr3], args=(Mirnv_B_exp_corr, R_filaments, z_filaments, R_mirnov, z_mirnov), consargs=None, rhobeg=0.1, rhoend=1e-10, maxfun=20000, catol=1e-10 )
		end = timer()
		print ("Time for corrected optimization: "+str(end-start))

		
		np.save('f_opt_7Filaments_7degree_'+str(SHOT_NUMBER)+'.npy',f_opt)
		np.save('f_opt_corr_7Filaments_7degree_'+str(SHOT_NUMBER)+'.npy',f_opt_corr)
	else:
		print("Loading data from shot number: "+str(SHOT_NUMBER)+"!")
		f_opt = np.load('f_opt_7Filaments_7degree_'+str(SHOT_NUMBER)+'.npy')
		f_opt_corr = np.load('f_opt_corr_7Filaments_7degree_'+str(SHOT_NUMBER)+'.npy')
	
	#Lets check how close is our minimization values to the experimental ones by applaying Biot-Savart with them 
	Mirnov_flux_experimental_multi = np.array([], np.float32)
	Mirnov_flux_corr_experimental_multi = np.array([], np.float32)
	

	Mirnov_flux_experimental_multi = np.append( Mirnov_flux_experimental_multi, BmagnMultiModule(R_filaments[0], z_filaments[0], f_opt[0:7], R_filaments, z_filaments, R_mirnov, z_mirnov ) )
	Mirnov_flux_corr_experimental_multi = np.append( Mirnov_flux_corr_experimental_multi, BmagnMultiModule(R_filaments[0], z_filaments[0], f_opt_corr[0:7], R_filaments, z_filaments, R_mirnov, z_mirnov) )

	#compute de error
	RMSE = np.sqrt( np.mean( np.square( np.subtract(Mirnov_flux_experimental_multi, Mirnv_B_exp ) ) ) )
	RMSE_corr = np.sqrt( np.mean( np.square( np.subtract(Mirnov_flux_corr_experimental_multi, Mirnv_B_exp_corr ) ) ) )
	
	# Matrix whose elements gives the contribution  to the measuremnt i  to a unitary current in the filament j [T]
	Mfp = np.zeros((12,7), np.float32)
	from function import Bmagnmirnv #def Bmagnmirnv( Z_filament,R_filament,I_filament,r_mirnv,z_mirnv):
	for i in range(12):
		for j in range(7):
			Mfp[i][j] = Bmagnmirnv(z_filaments[j], R_filaments[j], 1, R_mirnov[i], z_mirnov[i])
	start = timer()		
	Mpf = np.linalg.pinv(Mfp, rcond=1e-20)
	end = timer()
	print ("Time for pinv: "+str(end-start))
	I_Mpf_corr = Mpf.dot(Mirnv_B_exp_corr)
	
	#I_Mpf_corr = [-8433.61525597059, 656.265783834629, 1318.3380915753, 1278.5770300308, 1734.90770498183, 1501.37653118269, 1390.34150772538]

	Mirnov_flux_corr_theo_multi = np.array([], np.float32)
	Mirnov_flux_corr_theo_multi = np.append( Mirnov_flux_corr_theo_multi, BmagnMultiModule(R_filaments[0], z_filaments[0], I_Mpf_corr, R_filaments, z_filaments, R_mirnov, z_mirnov) )
	RMSE_theo_corr = np.sqrt( np.mean( np.square( np.subtract(Mirnov_flux_corr_theo_multi, Mirnv_B_exp_corr ) ) ) )
	#PLOTTIAMO
	
	plot.figure(20)
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp, '-o',label="Experimental Data")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_experimental_multi, '-*',label="Biot-savart(optimized )")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=195[ms]  Ip~4.1[kA] (Multifilament)")
	plot.legend([p1, p2])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	
	print("error with correct" + str(ErrorMirnFuncMultiFilam(f_opt_corr[0:7], Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov)))
	print("error without correct" + str(ErrorMirnFuncMultiFilam(f_opt[0:7], Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov)))
	print("error pseudi" + str(ErrorMirnFuncMultiFilam(I_Mpf_corr, Mirnv_B_exp, R_filaments, z_filaments, R_mirnov, z_mirnov)))	
	
	plot.figure(30)
	p1, = plot.plot(range(1, 13, 1), 1000*Mirnv_B_exp_corr, '-o',label="Experimental Data corrected (Multifilament)")
	p2, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_experimental_multi, '-*',label="Biot-savart(optimized )")
	p3, = plot.plot(range(1, 13, 1), 1000*Mirnov_flux_corr_theo_multi, '-s',label="theo multifilament mpf")
	plot.grid()
	plot.title("SHOT#"+str(SHOT_NUMBER)+" t=195[ms]  Ip~4.1[kA] (External flux corrected)")
	plot.legend([p1, p2, p3])
	plot.xlabel("Mirnov #")
	plot.ylabel("Optimization [mT]")
	plot.axis('equal')
	
	#Plasma, vessel and mirnov coil plot
	plot.figure(40)
	plot.plot(x_vessel,y_vessel,color='k', linewidth=2.0)
	plot.plot(46,0,'.m',MarkerSize = 480)
	plot.plot(R_filaments[0],z_filaments[0],'.k',MarkerSize = 10)
	plot.plot(R_filaments[0],z_filaments[0],'.b',MarkerSize = 20)
	for i in range(1,7):
		plot.plot(R_filaments[i],z_filaments[i],'.b',MarkerSize = 20)
	for i in range(12):
		plot.text(R_mirnov[i], z_mirnov[i], str(i+1), horizontalalignment='center', verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
	
	plot.text(57,0,'LFS',FontSize = 15)
	plot.text(33,0,'HFS',FontSize = 15)
	plot.xlabel('R[cm]')
	plot.ylabel('Z[cm]')
	plot.grid()
	plot.axis('equal')
	
	return [Mirnov_flux, Mirnov_flux_corr, f_opt, f_opt_corr, Mirnov_flux_experimental_multi, Mirnov_flux_corr_experimental_multi, RMSE, RMSE_corr, RMSE_theo_corr, Mpf, Mirnv_B_exp_corr, I_Mpf_corr, Mfp]