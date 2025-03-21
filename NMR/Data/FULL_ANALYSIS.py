# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:36:45 2024

@author: Admin
"""

import pandas as pd
import toolkit as tk

#for i in range(0,5):
#    exec(open('voltage_sweep.py').read())
    
file_name = "VOLTAGE SWEEP RAW [2024-04-04-11-36].csv"
#file_name_2 = "VOLTAGE SWEEP RAW [2024-04-04-12-07].csv"
df = pd.read_csv(file_name)
#df2 = pd.read_csv(file_name_2)


#df = df = df[df['Volts(V)']<=5]
frequency = df["Freq(Hz)"]
voltage = df["Volts(V)"]
ufreq = [20]*len(df)
uvolts = [0.05]*len(df)

#frequency2 = df2["Freq(Hz)"]
#voltage2 = df2["Volts(V)"]
#ufreq2 = df2["u(Freq)(Hz)"]

tk.block_print(
    [f"Freq: {frequency}",
     f"Volts: {voltage}",
     f"Uncert: {ufreq}",
     ]
    , "AHHHHH")

data = tk.curve_fit_data(voltage, frequency, fit_type='linear-int', res=True,
                  chi=True, uncertainty=ufreq)
#data2 = tk.curve_fit_data(voltage2, frequency2, fit_type='linear-int', res=True,
#                  chi=True, uncertainty=ufreq2)

#tk.quick_plot(voltage, frequency, uncertainty=ufreq)

meta = {"title" : "Voltage Frequency Analysis",
    "xlabel" : "Voltage (V)",
    "ylabel" : "Frequency (Hz)",
    'chi_sq' : data['chi-sq'],
    'fit-label': r"$f = m\cdot V + c$",
    'data-label': "Raw Data",
    'save-name' : 'volt-freq {current_time}',
    'loc' : 'lower right'}

meta2 = {"title" : "Voltage Frequency Analysis",
    "xlabel" : "Voltage (V)",
    "ylabel" : "Frequency (Hz)",
    'chi_sq' : 0,
    'fit-label': r"$f = m\cdot V + c$",
    'data-label': "Data",
    'save-name' : 'volt-freq {current_time}',
    'loc' : 'lower right'}

tk.quick_plot_residuals(voltage, frequency, 
                        data['graph-horz'], data['graph-vert'], 
                        residuals=data['residuals'], uncertainty=ufreq,
                        meta=meta, save=True,
                        uncertainty_x=uvolts)

#tk.quick_plot_residuals(voltage2, frequency2, 
#                        data['graph-horz'], data['graph-vert'], 
#                        residuals=data['residuals'], uncertainty=ufreq2,
#                        meta=meta, save=True)

tk.block_print(
    [f"CHI : {data['chi-sq']}",
     f"m : {data['popt'][0]} +/- {data['pstd'][0]}",
     f"c : {data['popt'][1]} +/- {data['pstd'][1]}"
     ]
    , "Best Fit Parameters and Fit Analysis")

#tk.block_print(
#    [f"CHI : {data2['chi-sq']}",
#     f"m : {data2['popt'][0]} +/- {data2['pstd'][0]}",
#     f"c : {data2['popt'][1]} +/- {data2['pstd'][1]}"
#     ]
#    , "Best Fit Parameters and Fit Analysis 2")