import matplotlib.pyplot as plot
from function import *
from CentroidPosition_nFilaments_correct import centroidPosition_N_filaments, centroidPosition_N_filaments_change_radius
from CentroidPosition_7Filament_correct import centroidPosition_7_filaments
import numpy as np
from scipy.optimize import fmin 
from numpy import pi
from valueOnFlatTop import valuOnFlatTop
from valueOnFlatTop_big import valuOnFlatTop_big
import matplotlib.animation as animation


plot.close("all")
SHOT_NUMBER = 45994

data_mirnv_corr, data_mirnv_corr_flux, Ip_magn_corr_value, times, sumIfil_value, r0_probes_value, z0_probes_value, r0_corr_value, z0_corr_value = getDataFromDatabase(SHOT_NUMBER)
time = 1e-03 * times #TIME IN ms
SavePlot (time,r0_corr_value,'Real_Time_Reconstruction_r0','Time[ms]','R[m]',SHOT_NUMBER)
SavePlot (time,z0_corr_value,'Real_Time_Reconstruction_z0','Time[ms]','Z[m]',SHOT_NUMBER)
#Mfp, Mpf, I_Mpf_corr = centroidPosition_N_filaments(12, times, data_mirnv_corr, data_mirnv_corr_flux, SHOT_NUMBER, False, Ip_magn_corr_value, sumIfil_value)

#f_opt, f_opt_corr, f_opt_constr, f_opt_constr_corr, Mirnov_flux_experimental_multi_constr, Mirnov_flux_corr_experimental_multi_constr, RMSE_const_corr, RMSE_const = centroidPosition_7_filaments(times, data_mirnv_corr, data_mirnv_corr_flux, SHOT_NUMBER, False, Ip_magn_corr_value)