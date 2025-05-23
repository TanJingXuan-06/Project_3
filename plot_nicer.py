import serial
from serial.tools import list_ports
import time 
import matplotlib.pyplot as plt
import csv 
import numpy as np 
from scipy import integrate

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from main import moving_avg_filter, baseline_als

    
def plot_together(time_pressure,pressure_list, time_band,band_list,figname,fig_num = 1) : 
    fig ,ax1 = plt.subplots(figsize=(10,2))
    ax2 = ax1.twinx()
    pressure_line, = ax1.plot(time_pressure,pressure_list,color='blue')
    plt.title(f"Inverted Pressure Data & Band Data Against Time")
    ax1.set_ylabel("Inverted Pressure")
    ax1.set_xlabel("Time (s)")
    plt.xlim([0,10])
    band_line, = ax2.plot(time_band,band_list,color='red') 
    ax2.set_ylabel("Band") 
    
    # Combine legends from both axes
    lines = [pressure_line, band_line]
    labels = ["Pressure", "Band"]
    ax1.legend(lines, labels, loc='best')
    
    
    # ax1.grid(axis='x')
    # ax2.grid(axis='x')
    
    
def plot_combined(combined_time,combined_data,fignum,title):
    plt.figure(fignum,figsize=(10,2))
    plt.plot(combined_time,combined_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Data")
    plt.grid()
    plt.xlim([0,10])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title(f"Combined Pressure and Band Data Against Time {title}")

def plot(time_pressure,pressure_list, time_band,title,band_list,fig_num = 1) : 
    plt.figure(fig_num,figsize=(10,2))
    plt.subplot(1,2,1)
    plt.plot(time_pressure,pressure_list,color='blue')
    plt.title(f"Pressure Data ({title})")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure")
    plt.xlim([0,10])
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(time_band,band_list,color='red')
    plt.title(f"Band Data ({title})")    
    plt.xlabel("Time (s)")
    plt.ylabel("Band")
    plt.xlim([0,10])
    plt.ylim([-3,3])
    plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Increase vertical space between subplots


def main() : 
    
    band_data = []
    pressure_data = []
    num = 0
    
    with open("Project_3/csv_files/band_data.csv", 'r', newline='') as file:
        reader = csv.reader(file)
        
        for line in reader :
            
        
            num += 1 
            if num == 4 : 
                for i in line:
                    band_data.append(float(i))
                    
            elif num > 4 : 
                break 
            
    num = 0
    
    with open("Project_3/csv_files/pressure_data.csv", 'r', newline='') as file:
        reader = csv.reader(file)
        
        for line in reader :
            
        
            num += 1 
            if num == 4 : 
                for i in line:
                    pressure_data.append(float(i))
                    
            elif num > 4 : 
                break 

    time_band = np.linspace(0,60 , len(band_data) )
    time_pressure = np.linspace(0,60, len(pressure_data) )
    
    band_x = 0 
    pressure_x = 0 
    
    for idx,i in enumerate(time_band) : 
        if i >= 10:
            band_x = idx
            print(band_x)
            break 
        
    for idx,i in enumerate(time_pressure) : 
        if i >= 10:
            pressure_x = idx
            print(pressure_x)
            break 
    
    time_pressure = time_pressure[0:pressure_x+1]
    pressure_data = pressure_data[0:pressure_x+1]
    time_band = time_band[0:band_x+1]
    band_data = band_data[0:band_x+1]
     
    # plot(time_pressure,pressure_data, time_band,"RAW",band_data,fig_num = 1)
    
    # Noise Reduction digiting signal processing 
    mavg_pressure_list = moving_avg_filter(pressure_data)
    mavg_band_list = moving_avg_filter(band_data)
    mavg_band_list = moving_avg_filter(mavg_band_list)
    time_pressure = np.linspace(0,10 , len(mavg_pressure_list) )
    time_band= np.linspace(0,10 , len(mavg_band_list) )
    # plot(time_pressure,mavg_pressure_list, time_band,"MAVG Filtered",mavg_band_list,fig_num = 2)
    
    mean_band = np.mean(mavg_band_list)
    std_band = np.std(mavg_band_list)
    mavg_band_list_norm = (mavg_band_list - mean_band) / std_band
    
    mean_pressure = np.mean(mavg_pressure_list)
    std_presure = np.std(mavg_pressure_list)
    mavg_pressure_list_norm = (mavg_pressure_list - mean_pressure) / std_presure
    time_pressure = np.linspace(0, 10, len(mavg_pressure_list_norm))
    time_band = np.linspace(0, 10, len(mavg_band_list_norm))
    
    # plot(time_pressure,mavg_pressure_list_norm, time_band,"Filtered + Normalized",mavg_band_list_norm,fig_num = 3)
    # 
    # plot_together(time_pressure, -mavg_pressure_list_norm, time_band, mavg_band_list_norm,figname = "MAVG_NORM_TGT",fig_num=4)
    
    combined_data = []

    
    if len(mavg_band_list_norm) > len(mavg_pressure_list_norm) : 
        
        for idx, i in enumerate(mavg_pressure_list_norm) : 
            
            combined_data.append((-i + mavg_band_list_norm[idx])/2)
            combined_time = time_pressure
        
    else : 
            
        for idx, i in enumerate(mavg_band_list_norm) : 
        
            combined_data.append((i - (mavg_pressure_list_norm[idx]))/2)
            combined_time = time_band
    
    plot_combined(combined_time,combined_data,5,"Before Correction")
    z = baseline_als(combined_data,lam=1e5,p =0.01)
    corrected_combined_data = combined_data - z 
    plot_combined(combined_time,corrected_combined_data,6,"After Correction")
    plt.show()       
    
if __name__ == "__main__":
    main()   