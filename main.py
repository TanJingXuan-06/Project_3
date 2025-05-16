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

TIME = 30 # define time 

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

        # take data according to a time frame we set 
        while time_diff <= TIME : 
            
            data_received = ser.readline().decode("utf-8").strip()    
            # print(data_received)      
            unprocessed_data = data_received.split("\n")

            
            if "Sensor 1: " in unprocessed_data[0]: 
                pressure = int(unprocessed_data[0].split(": ")[1])
                pressure_list.append(pressure)
                print(f"Sensor 1 : {pressure}")
    
                
            elif "Sensor 2: " in unprocessed_data[0]: 
                band = int(unprocessed_data[0].split(": ")[1])
                band_list.append(band)
                print(f"Sensor 2 : {band}")
                
            end_time = time.time()
            time_diff = end_time - start_time
 

                        
        return pressure_list , band_list 
                 
    except (KeyboardInterrupt, serial.SerialException) as e:
        print("Error") 
        
    ser.write(b"STP")       #"Send RUN to STM32"
    


def plot(time_pressure,pressure_list, time_band,band_list) : 
    plt.subplot(2,1,1)
    plt.plot(time_pressure,pressure_list)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(time_band,band_list)
    plt.grid()
    plt.show()
    
    
def moving_avg_filter(data_list) : 
    moving_avg_data_list = []
    window_size = 20
    
    for i in range(len(data_list) - window_size + 1):
        window_average = round(np.sum(data_list[i:i+window_size]) / window_size, 2)
        
        moving_avg_data_list.append(window_average)
    
    return moving_avg_data_list

def main(): 
    pressure_list, band_list = gather_data()
    time_pressure = np.linspace(0,len(pressure_list) -1 , len(pressure_list) )
    time_band= np.linspace(0,len(band_list) -1 , len(band_list) )
    plt.figure(1,figsize = (10,5))
    plot(time_pressure, pressure_list, time_band, band_list)
    
    mavg_pressure_list = moving_avg_filter(pressure_list)
    mavg_band_list = moving_avg_filter(band_list)
    mavg_band_list = moving_avg_filter(mavg_band_list)
    time_pressure = np.linspace(0,len(mavg_pressure_list) -1 , len(mavg_pressure_list) )
    time_band= np.linspace(0,len(mavg_band_list) -1 , len(mavg_band_list) )
    plot(time_pressure, mavg_pressure_list, time_band, mavg_band_list)
       
        
    
    
if __name__ == "__main__":
    main()

