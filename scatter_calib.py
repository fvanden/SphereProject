# -*- coding: utf-8 -*-
#################
import os
import datetime
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib  import cm
#################
"""
Module containing functions to create scatter plot and interpolated surface 
plot of the sphere calibration data

Created on Wed Oct 19 12:01:19 2016

@author: fvanden
"""
# for 75 m resolution: 16 to 20
# for 30 m resolution: 39 to 44
# for 15 m resolution: 78 to 85  

def simpleScatter(filename, polvar = 'Zh', range_g = None):
    """
    Creates a scatter plot of the sphere calibration data
    
    Parameters
    ----------
    filename, str : complete file and path name of the file
    
    polvar, str : polarimetric variable to plot
    
    range_g, int : gate to plot, if None, multiple gates within the range
        (radar-sphere) will be plotted (depends on resolution)
        
    Returns
    -------
    None
    """

    # read data #
    
    file_handle=Dataset(filename,'r')
    
    azimuth = file_handle.variables['Azimuth'][:]
    elevation = file_handle.variables['Elevation'][:]
    range_r = file_handle.variables['Range'][:]
    
    if polvar == 'Zdr':
        data1 = file_handle.variables['Zh'][:] 
        data2 = file_handle.variables['Zv'][:] 
        data = data1 - data2
    else:
        try:
            data = file_handle.variables[polvar][:] 
        except KeyError:
            print( ("%s does not exist in file")%(polvar) )
            raise
    # determine max and min gates
    
    resolution = range_r[1] - range_r[0]
    
    if resolution > 70:
        min_gate = 16
        max_gate = 20
    elif resolution > 25:
        min_gate = 39
        max_gate = 44
    else:
        min_gate = 80
        max_gate = 90
        
    try:
        vmin = valuesdict[polvar][0]
        vmax = valuesdict[polvar][1]
    except KeyError:
        vmin = np.nanmin(data)
        vmax = np.nanmin(data)
        print(vmin, vmax)
    
    # plot data #

    plt.ion()    
    
    if range_g is None:     
    
        for r in range(min_gate,max_gate):
            plt.figure()
            plt.scatter(azimuth,elevation,c=data[r], marker = 's', cmap = cm.jet, vmin = vmin, vmax =vmax)
            cbar = plt.colorbar()
            cbar.set_label(polvar)
            plt.title('range' + str(range_r[r]))
            plt.xlabel('Azimuth')
            plt.ylabel('Elevation')
    else:
        plt.figure()
        plt.scatter(azimuth,elevation,c=data[range_g], marker = 's', cmap = cm.jet, vmin = vmin, vmax =vmax)
        cbar = plt.colorbar()
        cbar.set_label(polvar)
        plt.title('range' + str(range_r[range_g]))
        plt.xlabel('Azimuth')
        plt.ylabel('Elevation')

def calibInterp(filename, polvar = 'Zh', gate = None, res_azim_interp = 0.4, res_elev_interp = 0.5):
    """
    Interpolates sphere calibration to a regular grid (very simple nearest
    neighbour interpolation, requires amelioration) and plots a contour plot
    of the sphere
    
    Parameters
    ----------
    filename, str : complete path and filename of the data to plot
    
    polvar, str : polarimetric variable to plot
    
    gate, int : range gate for which to plot data (if None, gate will be 
        decided based on resolution and experience (radar-sphere))
    
    res_azim_interp, float : interpolation resolution in degrees in azimuth
        direction, some options/combinations are given below 
        
    res_elev_interp, float : interpolation resolution in degrees in elevation
        direction, some options/combinations are given below
    
    res_azim_interp = 0.1
    res_elev_interp = 0.12
    
    res_azim_interp = 0.2
    res_elev_interp = 0.25
    
    res_azim_interp = 0.8
    res_elev_interp = 1.0
    
    Returns
    -------
    None
    
    Comments
    --------
    Use Python interpolation for nearest neighbour? Current technique assumes
    very small holes in data..
    
    #file = 'MXPol-polar-20160824-070335-WINDOW-013_5.nc'
    
    """
    
    file = os.path.basename(filename)

    # loading data
    
    file_handle=Dataset(filename,'r')
        
    azimuth = file_handle.variables['Azimuth'][:]
    elevation = file_handle.variables['Elevation'][:]
    range_r = file_handle.variables['Range'][:]
        
    data = file_handle.variables[polvar][:] 
    
    # determine resolution
    
    resolution = range_r[1] - range_r[0]

    
    if resolution > 70:
        r = 17
        resolution = 75
    elif resolution > 25:
        r = 42
        resolution = 30
    else:
        r = 86
        resolution = 15
            
    if gate is not None: 
        r = gate
        
        
    try:    
        vmin = valuesdict[polvar][0]
        vmax = valuesdict[polvar][1]
    except KeyError:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        print(vmin, vmax)
        
    data = data[r]
    res_text = str(int(resolution))
    range_obs = str('%.2f' %range_r[r])
    
    # determine date
    
    date, time = file.split('-')[2:4]
    date = datetime.datetime.strptime(date, '%Y%m%d')
    date = datetime.datetime.strftime(date, '%Y-%m-%d')
    
    time = datetime.datetime.strptime(time, '%H%M%S')
    time = datetime.datetime.strftime(time, '%H:%M:%S')
    
    # interpolation parameters
        
    azim_interp = np.arange(min(azimuth), max(azimuth), res_azim_interp)
    elev_interp = np.arange(min(elevation), max(elevation),res_elev_interp)
        
    data_grid = np.zeros((len(elev_interp), len(azim_interp)))
    num_samples = np.zeros((len(elev_interp), len(azim_interp)))
    
    # convert data to linear values for interpolation
    
    #data = Conversion().logToLin(data, polvar)
    
    
    # interpolate to grid
        
    for dp in range(0,len(data)):
        idx = ( np.abs(elev_interp - elevation[dp]).argmin(), np.abs(azim_interp - azimuth[dp]).argmin() )
        num_samples[idx]=num_samples[idx]+1 
        data_grid[idx]=data_grid[idx]+data[dp]
        
    num_samples[num_samples==0]=float('nan')                         
    data_grid=data_grid/num_samples
    
    """
    # interpolation Python
    
    X,Y = np.meshgrid(azim_interp, elev_interp)
    points = np.vstack( (X.ravel(), Y.ravel()) ).T
    data_interp = griddata(points, data_grid.ravel(), (X, Y), method = method)
    """
    
    # assign value of closest bin to missing pixels
    
    N = len(elev_interp)
    M = len(azim_interp)
    
    for i in range(0,N):
        for j in range(0,M):
            if (np.isnan(data_grid[i,j])):
                values = []
                if i-1 >= 0:
                    values.append(data_grid[i-1,j]) # pixel above
                if j-1 >= 0:
                    values.append(data_grid[i,j-1]) # pixel to left
                if i+1 < N:
                    values.append(data_grid[i+1,j]) # pixel below
                if j+1 < M:
                    values.append(data_grid[i,j+1]) # pixel to the right
                    
                data_grid[i,j] = np.nanmean(values)
                
    #data_grid = Conversion().linToLog(data_grid, polvar)
                
    # find max value
                
    max_val =  str('%.2f' %data_grid.max())
    mytext = 'max data: ' + max_val + '\n' + 'resolution: ' + res_text + ' m' + '\n' +\
    'Range: ' + range_obs + '\n' +\
    'Date: ' + date + '\n' + 'Time: ' + time + '\n'
    
    # plot
    
    X,Y = np.meshgrid(azim_interp, elev_interp)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.text(0.2, 0.85,mytext,
         fontdict = {'color':'w', 'weight': 'bold', 'fontsize' : 12, 'multialignment': 'left'},
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)
    
    CS = plt.contourf(X,Y,data_grid, cmap = cm.jet, vmin = vmin, vmax =vmax)
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.title(file)
    cbar = plt.colorbar()
    cbar.set_label(polvar, rotation=90)
    
    CS2 = plt.contour(CS, levels=CS.levels,
                      colors='k',
                      linewidths=(1,),
                      linestyle = '--',
                      origin='lower',
                      hold='on')
    plt.clabel(CS2, fmt='%2.1f', colors='k', fontsize=14)

    
valuesdict = {
    'Zh':[-10.0,40.0],
    'Zv':[-10.0,40.0],
    'Zdr':[-4.0,6.0], 
    'Sw':[0.0, 13]}

