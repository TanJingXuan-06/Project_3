# import 
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

TIME = 60 # define time 

# the target string to determine if STM 32 is connected 
STM32Name = "STMicroelectronics STLink Virtual COM Port"

def check_STM32_connectivity():
    """ 
    Attempts to find and connect to the STM32.
    
    Returns: 
        ListPortInfo: The STM32 port. 
        int: -1 if a port isn't found.
    """ 
    # get the list of ports
    listOfPorts = list_ports.comports()
    # loop through all the items in the list 
    for indexList in range (len(listOfPorts)) : 
        port = listOfPorts[indexList]   
        # convert the portName to string to use find()
        portNameStr = str(port)
        # find the index of the STM32 
        # if not found will return -1 
        if portNameStr.find(STM32Name) != -1 : 
            stm32_port = port
            return stm32_port 
            
        
    return -1

def gather_data() : 
    # input : input from user to get data 
    # output : start stm 
    """ 
    Records and prints voltage reading from the sensor
    Gets 20 raw voltage data from the sensor including the mean of these 20 datas. 

    Returns: 
        Mean data received : if STM32 board is found.
        None: if STM32 board isn't found.
    """ 
    stm32_port = check_STM32_connectivity()
    pressure_list = []
    band_list     = []
    
    print(" ------------------------------------ GATHER DATA ------------------------------------")
    
    if stm32_port == -1 :
        print("STM32 board not found. Please ensure it is connected.")
        return 
    
    try : 
        ser = serial.Serial(port =stm32_port.name, baudrate=115200 , timeout=1)    
        print("Reading from STM.....")
        time.sleep(0.5)
        
        # calculate time 
        start_time = time.time() 
        end_time = time.time()
        time_diff = end_time - start_time
        pressure = 0 
        band = 0

        # take data according to a time frame we set 
        while time_diff <= TIME : 
            
            data_received = ser.readline().decode("utf-8").strip()    
            # print(data_received)      
            unprocessed_data = data_received.split("\n")

            
            if "Sensor 1: " in unprocessed_data[0]: 
                pressure = int(unprocessed_data[0].split(": ")[1])
                pressure_list.append(pressure)
                # print(f"Sensor 1 : {pressure}")
    
                
            elif "Sensor 2: " in unprocessed_data[0]: 
                band = int(unprocessed_data[0].split(": ")[1])
                band_list.append(band)
                
                
            end_time = time.time()
            time_diff = end_time - start_time
            print(f"TIME: {round(time_diff,3)}\t\tSensor 1 : {pressure}\t\tSensor 2 : {band}")
 

                        
        return pressure_list , band_list 
                 
    except (KeyboardInterrupt, serial.SerialException) as e:
        print("Error") 
        
    ser.write(b"STP")       #"Send RUN to STM32"
    
def plot(time_pressure,pressure_list, time_band,band_list,figname,fig_num = 1) : 
    plt.figure(fig_num,figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(time_pressure,pressure_list,color='blue')
    plt.title(f"Pressure Data ({figname})")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.xlim([0,TIME])
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(time_band,band_list,color='red')
    plt.title(f"Band Data ({figname})")    
    plt.xlabel("Time")
    plt.ylabel("Band")
    plt.xlim([0,TIME])
    plt.grid()
    plt.subplots_adjust(hspace=0.5)  # Increase vertical space between subplots
    
def plot_together(time_pressure,pressure_list, time_band,band_list,figname,fig_num = 1) : 
    fig ,ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()
    ax1.plot(time_pressure,pressure_list,color='blue')
    plt.title(f"Pressure Data/Band Data ({figname})")
    ax1.set_ylabel("Pressure")
    ax1.set_xlabel("Time")
    plt.xlim([0,TIME])
    ax2.plot(time_band,band_list,color='red') 
    ax2.set_ylabel("Band")        
    plt.grid()
 
def plot_combined(combined_time,combined_data,fignum,title):
    plt.figure(fignum)
    plt.plot(combined_time,combined_data)
    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.grid()
    plt.title(f"Combined Data Plot {title}")
    plt.xlim([0,TIME])
     
def moving_avg_filter(data_list) : 
    moving_avg_data_list = []
    window_size = 20
    
    for i in range(len(data_list) - window_size + 1):
        window_average = round(np.sum(data_list[i:i+window_size]) / window_size, 2)
        
        moving_avg_data_list.append(window_average)
    
    return moving_avg_data_list

def pred(data_list,thres): 
    
    valley_indices, ___ = find_peaks(data_list, distance=50, prominence=thres)
    
    return len(valley_indices), valley_indices

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z


def main(): 
    
    
    while True : 
        
        user = input("Y/N")  
        if user == "Y" or user == 'y' : 
            # gather data 
            pressure_list, band_list = gather_data()
            
            # plotting 
            time_pressure = np.linspace(0,TIME , len(pressure_list) )
            time_band= np.linspace(0,TIME , len(band_list) )
            plot(time_pressure, pressure_list, time_band, band_list, figname = "Raw",fig_num=1)
            
            # Noise Reduction digiting signal processing 
            mavg_pressure_list = moving_avg_filter(pressure_list)
            mavg_band_list = moving_avg_filter(band_list)
            mavg_band_list = moving_avg_filter(mavg_band_list)
            time_pressure = np.linspace(0,TIME , len(mavg_pressure_list) )
            time_band= np.linspace(0,TIME , len(mavg_band_list) )
            plot(time_pressure, mavg_pressure_list, time_band, mavg_band_list,figname = "MAVG",fig_num=2)
            
            mean_band = np.mean(mavg_band_list)
            std_band = np.std(mavg_band_list)
            mavg_band_list_norm = (mavg_band_list - mean_band) / std_band
            
            mean_pressure = np.mean(mavg_pressure_list)
            std_presure = np.std(mavg_pressure_list)
            mavg_pressure_list_norm = (mavg_pressure_list - mean_pressure) / std_presure
            time_pressure = np.linspace(0, TIME, len(mavg_pressure_list_norm))
            time_band = np.linspace(0, TIME, len(mavg_band_list_norm))


            ## Data Processing     
            band_pred, ___ = pred((np.array(mavg_band_list_norm)),0.2)
            pressure_pred, ___ = pred(-mavg_pressure_list_norm,0.3)
            
            # print(f"Pressure Pred: {pressure_pred} \nBand pred: {band_pred}")
            plot(time_pressure, mavg_pressure_list_norm, time_band, mavg_band_list_norm,figname = "MAVG_NORM",fig_num=3)          
            
            plot_together(time_pressure, -mavg_pressure_list_norm, time_band, mavg_band_list_norm,figname = "MAVG_NORM_TGT",fig_num=4)
            
            combined_data = []

            
            if len(mavg_band_list_norm) > len(mavg_pressure_list_norm) : 
                
                for idx, i in enumerate(mavg_pressure_list_norm) : 
                    
                    combined_data.append((-i + mavg_band_list_norm[idx])/2)
                    combined_time = time_pressure
                    
            else : 
                 
                 for idx, i in enumerate(mavg_band_list_norm) : 
                    
                    combined_data.append((i - (mavg_pressure_list_norm[idx]))/2)
                    combined_time = time_band
                    
            combined_pred,___ = pred(combined_data,1)
            plot_combined(combined_time,combined_data,5,"Before Correction")
            print(f"Breathe Rate (Combined Pred): {combined_pred}")
            
            
            
            z = baseline_als(combined_data,lam=1e5,p =0.01)
            corrected_combined_data = combined_data - z 
            plot_combined(combined_time,corrected_combined_data,6,"After Correction")
            
            corrected_combined_pred,___ = pred(corrected_combined_data,0.9)
            print(f"Breathe Rate (Corrected Combined Pred): {corrected_combined_pred}")
            plt.show() 
                        
            # with open("Project_3/combined_data.csv", 'w', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(combined_data)
                
            # with open("Project_3/corrected_combined_data.csv", 'w', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(corrected_combined_data)
                                
             
        elif user == "N" or user == "n" : 
            break
        
        else : 
            pass 
        
if __name__ == "__main__":
    main()