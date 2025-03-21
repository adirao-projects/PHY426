import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import toolkit as tk
from uncertainties import ufloat
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


def ideal_lens_eqn(pinv, finv):
    return finv - pinv

def ideal_lens_eqn_acc(p, f):
    return (1/f) - (1/p)
    
def thin_lens_eqn(pfpinv, finv, qbpinv):
    return finv - pfpinv + qbpinv


def load_data(path):
    df = pd.read_csv(path)
    
    
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
    

def fit_ideal_lens(pdata, qdata, mdata):
    meta = {'title':f'Fit to Ideal Lens Equation',
            'xlabel':'Object Distance',
            'ylabel':'Image Distance',
            'fit-label': r"$\frac{1}{q} = \frac{1}{f} - \frac{1}{p}$",
            'data-label': "Raw Data",
            'loc':'upper right',
            'save-name':'ideallens'}


    print(pdata)
    
    pinv = np.reciprocal(pdata)
    qinv = np.reciprocal(qdata)
    plt.errorbars(pinv, qinv, fmt='o', markersize='4', 
                  color='red', ecolor='black',
                   xerr=CALIPER_UNCERT, yerr=CALIPER_UNCERT, )

    pinv_u = np.sqrt((CALIPER_UNCERT*(pinv**2))**2)

    tk.quick_analyze(pinv, qinv, yerr=pinv_u, fit_type='custom',
                     model_function_custom = ideal_lens_eqn,
                      res=True, chi=True, meta = meta, 
                      show=True, save=True, dataout=True,
                      params=('A', r'\tau', 'x_0'), rounding=False)
    
    
    
def fit_ideal_lens_df(df):
    meta = {'title':'Fit to Ideal Lens Equation',
            'xlabel':r'$\frac{1}{p}\,\text{m}^{-1}$',
            'ylabel':r'$\frac{1}{q}\,\text{m}^{-1}$',
            'fit-label': r"$\frac{1}{q} = \frac{1}{f} - \frac{1}{p}$",
            'data-label': "Raw Data",
            'loc':'upper right',
            'save-name':'ideallens'}


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
    
    pdata = np.abs(lens - src)
    qdata = np.abs(lens - img)
    
    plt.figure()
    plt.title('Non inv')
    plt.errorbar(pdata, qdata, fmt='o', markersize='4', 
                  color='red', ecolor='black',
                   xerr=CALIPER_UNCERT, yerr=CALIPER_UNCERT, )
    plt.show()
    
    pinv = np.reciprocal(pdata)
    qinv = np.reciprocal(qdata)
    
    plt.figure()
    plt.title('inv')
    plt.errorbar(pinv, qinv, fmt='o', markersize='4', 
                  color='red', ecolor='black',
                   xerr=CALIPER_UNCERT, yerr=CALIPER_UNCERT, )
    plt.show()
    pinv_u = np.sqrt((CALIPER_UNCERT*(pinv**2))**2)

    data = tk.quick_analyze(pinv, qinv, yerr=pinv_u, fit_type='linear-int',
                     model_function_custom = ideal_lens_eqn,
                      res=True, chi=True, meta = meta, 
                      show=True, save=True, dataout=True,
                      params=('m', r'\frac{1}{f}'), rounding=False)

    print(data['popt'], data['pstd'])    
    f = data['popt'][1]
    fu = data['pstd'][1]
    
    fu = np.abs(fu*((1/f)**2))
    f = 1/f
    
    plt.figure()
    plt.title('Non inv')
    plt.errorbar(pdata, qdata, fmt='o', markersize='4', 
                  color='red', ecolor='black',
                   xerr=CALIPER_UNCERT, yerr=CALIPER_UNCERT, )
    qdata_pred = np.reciprocal(ideal_lens_eqn_acc(pdata, f))
    plt.plot(pdata, qdata_pred)
    plt.show()
    
    
    print(f, '+/-', fu)
    
def fit_ideal_lens(df):
    meta = {'title':'Fit to Ideal Lens Equation',
            'xlabel':r'$\frac{1}{p}\,\text{m}^{-1}$',
            'ylabel':r'$\frac{1}{q}\,\text{m}^{-1}$',
            'fit-label': r"$\frac{1}{q} = \frac{1}{f} - \frac{1}{p}$",
            'data-label': "Raw Data",
            'loc':'upper right',
            'save-name':'ideallens'}


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
    
    pdata = np.abs(lens - src)
    qdata = np.abs(lens - img)
    
    plt.figure()
    plt.title('Non inv')
    plt.errorbar(pdata, qdata, fmt='o', markersize='4', 
                  color='red', ecolor='black',
                   xerr=CALIPER_UNCERT, yerr=CALIPER_UNCERT, )
    plt.show()
    
    pinv = np.reciprocal(pdata)
    qinv = np.reciprocal(qdata)
    
    plt.figure()
    plt.title('inv')
    plt.errorbar(pinv, qinv, fmt='o', markersize='4', 
                  color='red', ecolor='black',
                   xerr=CALIPER_UNCERT, yerr=CALIPER_UNCERT, )
    plt.show()
    pinv_u = np.sqrt((CALIPER_UNCERT*(pinv**2))**2)

    data = tk.quick_analyze(pinv, qinv, yerr=pinv_u, fit_type='linear-int',
                     model_function_custom = ideal_lens_eqn,
                      res=True, chi=True, meta = meta, 
                      show=True, save=True, dataout=True,
                      params=('m', r'\frac{1}{f}'), rounding=False)

    print(data['popt'], data['pstd'])    
    f = data['popt'][1]
    fu = data['pstd'][1]
    
    fu = np.abs(fu*((1/f)**2))
    f = 1/f
    
    plt.figure()
    plt.title('Non inv')
    plt.errorbar(pdata, qdata, fmt='o', markersize='4', 
                  color='red', ecolor='black',
                   xerr=CALIPER_UNCERT, yerr=CALIPER_UNCERT, )
    qdata_pred = np.reciprocal(ideal_lens_eqn_acc(pdata, f))
    plt.plot(pdata, qdata_pred)
    plt.show()
    
    
    print(f, '+/-', fu)
    
    
if __name__ == '__main__':
    df = load_data('../Data/ExpA-2025.01.24-1.csv')
    df_11 = df[df['lens-code']==1.1]
    df_12 = df[df['lens-code']==1.2]
    df_13 = df[df['lens-code']==1.3]
    df_14 = df[df['lens-code']==1.4]
    df_14 = df[df['lens-code']==1.4]
    
    
    #print(df)
    out_11 = analyze(df_11, GRID_SIZE)
    out_12 = analyze(df_12, GRID_SIZE)
    out_13 = analyze(df_13, GRID_SIZE)
    out_14 = analyze(df_14, GRID_SIZE)
    
    
    pdata = np.array(out_11[0] + out_12[0] + out_13[0] + out_14[0])
    qdata = np.array(out_11[1] + out_12[1] + out_13[1] + out_14[1])
    mdata = np.array(out_11[2] + out_12[2] + out_13[2] + out_14[2])
    
    #fit_mag(pdata, qdata, mdata)
    #fit_thin_lens(pdata, qdata, mdata)
    
    #fit_ideal_lens(pdata, qdata, mdata)
    
    df = load_data('../Data/ExpA-2025.01.31-1.csv')
    df_ = df[df['lens-code']==3.1]
    fit_ideal_lens_df(df_11)
    
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
    