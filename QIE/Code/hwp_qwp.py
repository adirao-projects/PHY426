#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:47:01 2025

@author: Aditya K. Rao
@github: @adirao-projects
"""

import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', names=['Raw', 'Cor'],
                     skiprows=1)
    
    df['uRaw'] = df['Raw'].std()
    df['uCor'] = df['Cor'].std()
    
    df['Adj'] = df['Raw'] - df['Cor']
    
    return df

if __name__ == '__main__':
    # df_A90_B90 = load_data('../Data/03.21/N9090.txt')
    # df_A00_B00 = load_data('../Data/03.21/N0000.txt')
    
    df_A90_B90 = load_data('../Data/03.21/N9090_2.txt')
    df_A00_B00 = load_data('../Data/03.21/N00_2.txt')
    
    m9090 = df_A90_B90['Adj'].mean()
    m0000 = df_A00_B00['Adj'].mean()
    
    s9090 = df_A90_B90['Adj'].std()
    s0000 = df_A00_B00['Adj'].std()
    
    
    print(f'90 90 : {m9090} +/- {s9090}')
    print(f'00 00 : {m0000} +/- {s0000}')
    
