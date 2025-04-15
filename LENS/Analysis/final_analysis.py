import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from uncertainties import ufloat
import toolkit as tk

# Constants
# Note that we are measuring a 3x3 grid which is where the multiplication
# by 3 comes from
GRID_SIZE = 1 # mm
GRID_SIZE = GRID_SIZE*3

CALIPER_UNCERT = 0.01

# Measurement Uncertainty remains the same
GRID_SIZE_u = 0.5 #mm uncert

OFFSETS = {
    "LENS":59.95,
    'LENS-T':25.42,
    "IMG":59.82,
    'LAMP':90.74,
    'APP':59.82
    }


def thin_lens_eqn(p, finv):
    return finv - 1/p

def thin_lens_eqn2(p, finv, fp, bp):
    return (finv - 1/(p-fp))**(-1)+bp
    

def load_data(path):
    df = pd.read_csv(path)
    
    df['p'] = np.abs(df['lens'] - df['source'])
    df['q'] = np.abs(df['lens']-df['image'])
    
    return df


def analyze(df, grid_size):    

    df['Uncert'] = pd.Series([CALIPER_UNCERT for _ in range(len(df))])
    
    print(df)    

    
    img = [ufloat(i,CALIPER_UNCERT) for i in df['image'].to_numpy()]
    lens = [ufloat(i,CALIPER_UNCERT) for i in df['lens'].to_numpy()]
    src = [ufloat(i,CALIPER_UNCERT) for i in df['source'].to_numpy()]
    
    img = df['image'].to_numpy()
    lens = df['lens'].to_numpy()
    src =  df['source'].to_numpy()
    
    # Adding offsets
    img = img + 0.5*OFFSETS['IMG']*0.1
    lens = lens + 0.5*OFFSETS['LENS']*0.1 + OFFSETS['LENS-T']*0.1
    src = src - 0.5*OFFSETS['APP']
    
    #print(img, obj, lens)
    
    #mag_uncert = ufloat(df_mean['gridsize'], df_std['gridsize'])
    #gridsize_uncert = ufloat(GRID_SIZE, GRID_SIZE_u)
    
    p = np.abs(lens - src)
    q = np.abs(lens - img)
    
    #print(p,q)
    
    m = df['gridsize'].div(3)
    
    magnification = -(q/p)
    
    #magnification = mag_uncert/gridsize_uncert
   
    output =(p.tolist(),
             q.tolist(),
             m.tolist())
    
    return output
    
def fit_thin_lens(pvals, qvals, gvals):
    
    

if __name__ == '__main__':
    df = load_data('../Data/ExpA-2025.01.24-1.csv')
    df_11 = df[df['lens-code']==1.1]
    df_12 = df[df['lens-code']==1.2]
    df_13 = df[df['lens-code']==1.3]
    df_14 = df[df['lens-code']==1.4]
        
    pvals = []
    pvals_u = []
    qvals = []
    qvals_u = []
    gvals = []
    gvals_u = []
    for df_ in [df_11, df_12, df_13, df_14]:
        pvals.append(df_['p'].mean())
        pvals_u.append(df_['p'].std())
        qvals.append(df_['q'].mean())
        pvals_u.append(df_['q'].std())
        gvals.append(df_['gridsize'].mean())
        gvals_u.append(df_['gridsize'].std())
        
    
    xtest = np.arange(27,50, 0.1)
    ytest = thin_lens_eqn(xtest, f=5.6)
    
    plt.figure()
    plt.plot(xtest, ytest)
    plt.show()
    

    # fit_mag(pdata, qdata, mdata)
    fit_thin_lens(pdata, qdata, mdata)
    
    # df2 = load_data('../Data/ExpA-2025.01.31-1.csv')    
    # df_21 = df2[df2['lens-code']==2.1]
    # out_21 = analyze(df_21, GRID_SIZE)
    # pdata = np.array(out_21[0])
    # qdata = np.array(out_21[1])
    # mdata = np.array(out_21[2])
    # fit_mag(pdata, qdata, mdata)
    # fit_thin_lens(pdata, qdata, mdata)
    
    
    # df3 = load_data('../Data/ExpA-2025.01.31-1.csv')    
    # df_31 = df3[df3['lens-code']==3.1]
    # out_31 = analyze(df_31, GRID_SIZE)
    # pdata = np.array(out_31[0])
    # qdata = np.array(out_31[1])
    # mdata = np.array(out_31[2])
    # fit_mag(pdata, qdata, mdata)
    # fit_thin_lens(pdata, qdata, mdata)
    