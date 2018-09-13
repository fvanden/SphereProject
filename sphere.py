#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############
import numpy as np
import math
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import gridspec
from matplotlib  import cm
from config_pl import generate_polvar_metadata
from conversion import Conversion
import scipy.optimize as opt
from scipy.interpolate import griddata
import warnings
##############

"""
Module to create plots and perform gaussian fits on the sphere calibration data 

Created on Wed Oct 19 14:13:15 2016

@author: fvanden

Revision history:  19.10.2016 - created

"""

#filename = '/ltedata/Payerne_balloon_2016/Radar/Proc_data/2016/08/24/MXPol-polar-20160824-123033-WINDOW-011_6.nc'

## -------------------------------------------------------------------------- ##
class GaussSphere():
    """
    A class for fitting 1 and 2D gaussian functions to 
    the raw sphere calibration data
    
    Parameters
    ----------
    filename, str : name of the sphere calibration file
        i.e. '/ltedata/Payerne_balloon_2016/Radar/Proc_data/2016/08/24/
        MXPol-polar-20160824-123033-WINDOW-011_6.nc'
    
    cutdB, float : dB value at which to cut off data
    
    doa, float : offset from the centre from which to use the data
        in degrees azimuth, if None default of 2.5 degrees is used
        
    doe, float : offset from the centre from which to use the data
        in degrees elevation, if None default of 2.5 degrees is used
        
    gate, int : gate for which to extract data. If None, gate will be 
        selected based on the resolution of the scan (see the find_gate
        function)
        
    var, str : type of radar variable to use. If None, the horizontal
        signal is used (SignalH)    
    
    azim, float : azimuth at which to take middle of the sphere; if None, the
        highest measured value will be considered the middle
        
    elev, float : elevation at which to take middle of the sphere; if None, the
        highest measured value will be considered the middle
        
        
    Attributes
    ----------
    
    azimuth, list : list of all azimuths from file
    
    elevation, list : all elevations from file
    
    data, list : all data values from file
    
    info_fit, dict : contains estimated optimum and one standard deviation on
        error of the estimated optimum for the 1d and 2d gaussian fits. 
        
    res_azim_interp, float : the azimuth resolution to interpolate data to a
        regular grid
        
    res_elev_interp, float : the elevation resolution to interpolate data to a
        regular grid
        
    theta, float : initial value for clockwise rotation of the gaussian 
        function.
    
    sigx, float : initial value for spread of gaussian function in x direction
    
    sigy, float : initial value for spread of gaussian function in y direction
        
    resolution, int : measurement resolution (15,30 or 75)
    
    """
    ## ------------------------------------------------------------------ ##
    ## Constructors/Destructors                                           ##
    ## ------------------------------------------------------------------ ##
    
    def __init__(self, filename, cutdB = None, doa = None, doe = None, gate = None, var = None, 
                 azim = None, elev = None):
                     
        # --- input parameters --- #
        
        self.filename = filename
        
        if cutdB is None:
            self.cutdB = 10.
        else:
            self.cutdB = cutdB
        
        if doa is None:
            self.doa = 1.5
        else:
            self.doa = doa
            
        if doe is None:
            self.doe = 1.5
        else:
            self.doe = doe
            
        if var is None:
            self.var = 'SignalH'
        else:
            self.var = var
            
        self.load_data()
            
        if gate is None:
            self.gate = self.find_gate()
        else:
            self.gate = gate
            
        self.azim = azim
        self.elev = elev
        
        # --- fitting output --- #
            
        self.info_fit = {}
        
        # --- interpolation parameters --- #
        
        self.res_azim_interp = 0.16
        self.res_elev_interp = 0.12
        
        # --- fitting parameters --- #
        
        self.theta = 0.
        self.sigx = 1.
        self.sigy = 0.5
         
        
        
    def __del__(self):
        pass
    
    ## ------------------------------------------------------------------ ##
    ## Methods                                                            ##
    ## ------------------------------------------------------------------ ##
    
    # public:
    
    ## -------------------------Plot functions--------------------------- ##
    ## ------------------------------------------------------------------ ##
    
    def plot_all(self, plot = None):
        """
        Plots 1d and 2d plots of data in one figure
        
        Parameters
        ----------
        plot, str : "scatter" to plot data in scatter plot, "gauss" to plot
            data in 2D gaussian plot. If None, plots scatter plot by default
            
        NOTE: this is the 'new' plotting function which uses plot_gauss,
        gaus_data_rad and data_2drad functions; data is selected within a 
        spherical area of radius self.doa around the center, and filtered to
        take only values higher then self.cutdB. This data is used for the 
        2D gaussian fit, then interpolated to a spherical grid using linear
        interpolation.
            
        Returns
        -------
        None
        
        See Also
        --------
        plot_all2
        """
        fig = plt.figure(figsize = [9.45, 5.8] )
        gs = gridspec.GridSpec(3,4)
        
        # elevation axis
        ax1 = plt.subplot(gs[:2,0])
        self.plot_elev(ax = ax1)
        
        # azimuth axis
        ax2 = plt.subplot(gs[2,1:3])
        self.plot_azim(ax = ax2)
        
        # create 2D axis
        ax3 = plt.subplot(gs[:2, 1:3])
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        
        # create gauss info panel
        ax4 = plt.subplot(gs[-1,-1], frameon = False)
        ax4.axes.get_yaxis().set_visible(False)
        ax4.axes.get_xaxis().set_visible(False)
        
        # create and plot 2D plot
        
        if plot == 'scatter' or plot is None:
            self.plot_scatter(ax = ax3)
        else:
            self.plot_gauss(ax = ax3)
            
        # create and plot gauss info panel text
            
        if plot == 'scatter' or plot is None:
            axtext = self.create_fit_text(ttype = '1d')
        else:
            axtext = self.create_fit_text(ttype = '2d')
        
        ax4.text(0.5,-0.4, axtext, fontsize = 7)
    
    def plot_all2(self, plot = None):
        """
        Plots 1d and 2d plots of data in one figure
        
        Parameters
        ----------
        plot, str : "scatter" to plot data in scatter plot, "gauss" to plot
            data in 2D gaussian plot. If None, plots scatter plot by default
            
        NOTE: this is the 'old' plotting function which uses the plot_gauss2 
        function and the interp functions; data is first interpolated to a 
        regular grid using nearest neighbour interpolation, then the sphere is
        selected within a square frame based on self.doe and self.doa, then 
        fitted with a 2D gaussian function.
            
        Returns
        -------
        None
        
        See Also
        --------
        plot_all
        """
        fig = plt.figure(figsize = [9.45, 5.8] )
        gs = gridspec.GridSpec(3,4)
        
        # elevation axis
        ax1 = plt.subplot(gs[:2,0])
        self.plot_elev(ax = ax1)
        
        # azimuth axis
        ax2 = plt.subplot(gs[2,1:3])
        self.plot_azim(ax = ax2)
        
        # create 2D axis
        ax3 = plt.subplot(gs[:2, 1:3])
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        
        # create gauss info panel
        ax4 = plt.subplot(gs[-1,-1], frameon = False)
        ax4.axes.get_yaxis().set_visible(False)
        ax4.axes.get_xaxis().set_visible(False)
        
        # create and plot 2D plot
        
        if plot == 'scatter' or plot is None:
            self.plot_scatter(ax = ax3)
        else:
            self.plot_gauss2(ax = ax3)
            
        # create and plot gauss info panel text
            
        if plot == 'scatter' or plot is None:
            axtext = self.create_fit_text(ttype = '1d')
        else:
            axtext = self.create_fit_text(ttype = '2d')
        
        ax4.text(0.5,-0.4, axtext, fontsize = 7)
            
    def plot_selection(self, polvar = None, title = None):
        """
        plots a figure indicating the data which was selected to perform the
        2D gaussian fit (only valid for plot_all function)
        
        Parameters
        ----------
        polvar, str : polarimetric variable, if set to None, the self.var
            will be used
        
        Returns
        -------
        None
        """
        
        ind, select = self.data_2drad(polvar = polvar)
        
        if polvar is None:
            pldata = self.data[self.gate]
        else:
            pldata = self.file_handle.variables[polvar][self.gate]
            
        # find max min values
        if polvar is not None:
            metadata = generate_polvar_metadata(polvar)
            vmin = metadata['valid_min']
            vmax = metadata['valid_max']
            cbartitle = metadata['long_name']
        elif self.var == 'SignalH':
            vmin = min(self.data[self.gate])
            vmax = max(self.data[self.gate])
        
        elif self.var == 'Zh':
            vmin = -10.
            vmax = 40.
            cbartitle = 'Reflectivity [dBZ]'
        else:
            metadata = generate_polvar_metadata(self.var)
            vmin = metadata['valid_min']
            vmax = metadata['valid_max']
            cbartitle = metadata['long_name']
            
        fig = plt.figure()
        plt.hold(True)
        plt.scatter(self.azimuth, self.elevation, c=pldata, marker = 's', cmap = plt.cm.jet,
                    vmin = vmin, vmax = vmax)
        cbar = plt.colorbar()
        cbar.set_label(cbartitle)
        plt.scatter(np.take(self.azimuth, ind), np.take(self.elevation,ind), marker = 'o', 
                    facecolors='none', edgecolors='w') 
        plt.plot(self.azim_sphere, self.elev_sphere, color = 'k')
        
        plt.xlabel('Azimuth')
        plt.ticklabel_format(useOffset=False)
        plt.ylabel('Elevation')
        if title is None:
            plt.title( ("selected data for radius sphere: %f and maxval: %f")%(self.doa, self.cutdB) )
        else:
            plt.title(title)
            
        
    def plot_gauss(self, ax = None):
        """
        Plots a 2D gaussian plot from data, selecting first the data in a
        radius of self.doa around the estimated center of the sphere
        (self.data_2drad), then fitting the data to a 2D gaussian function and 
        finally interpolating the data to a speherical grid using linear
        interpolation (for plotting purposes).
        
        Parameters
        ----------
        ax, axis handle : if None, creates a new figure
        
        Returns
        -------
        None
        
        See Also
        --------
        plot_gauss2
        
        """
        # fit data to original data (selected with a radius of self.doa round
        # the estimated middle of the sphere)
        
        best_vals, covar = self.gauss_data_rad()
        
        # get azimuth and elevation data and create gaussian fitted data
        
        ind, select = self.data_2drad()
        
        newazim = np.take(self.azimuth, ind)
        newelev = np.take(self.elevation, ind)
        
        z = (newazim, newelev)
        
        data_fitted = self.gaussian_2d(z, *best_vals)
        
        # interpolate data for plotting purposes
        
        aziminterp, elevinterp, datainterp = self.interp3(newazim, newelev, np.asarray(select))
        _,_,data_fit_interp = self.interp3(newazim, newelev, data_fitted)
        
        # get metadata
        if self.var == 'SignalH':
            units = 'mW'
        else:
            metadata = generate_polvar_metadata(self.var)
            units = metadata['units']
        
        # plot
        
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.hold(True)
            c = ax.imshow(datainterp, cmap=plt.cm.jet, origin='bottom', 
                           extent=(aziminterp.min(), aziminterp.max(), elevinterp.min(), elevinterp.max()))
            CS = ax.contour(aziminterp, elevinterp, data_fit_interp, 6, colors='w')         
            plt.clabel(CS, fontsize = 10, inline = 1)
            cbar = plt.colorbar(c)
            cbar.set_label(self.var + ' ' + units)
            plt.title('range' + str(self.range_r[self.gate]))
            plt.xlabel('Azimuth')
            plt.ylabel('Elevation')
            ax.ticklabel_format(useOffset=False)
        else:
            ax.hold(True)
            c = ax.imshow(datainterp, cmap=plt.cm.jet, origin='bottom', 
                           extent=(aziminterp.min(), aziminterp.max(), elevinterp.min(), elevinterp.max()))
            box = ax.get_position()
            axc = plt.axes([box.x0*1.05 + box.width * 1.05, box.y0, 0.01, box.height])
            cbar = plt.colorbar(c, cax = axc)
            cbar.set_label(self.var + ' ' + units)
            CS = ax.contour(aziminterp, elevinterp, data_fit_interp, 6, colors='w')
            plt.clabel(CS, fontsize = 10, inline = 1)
        
    def plot_gauss2(self, ax = None):
        """
        Plots a 2D gaussian plot from the data, interpolating the data first 
        (self.interp), and selecting a square area around the sphere with 
        azimuth extending self.doa degrees from the estimated centre of the 
        sphere and elevation extending self.doe from the estimated centre of 
        the sphere (self.data_2d). The data is then fitted to a 2D gaussian 
        function (self.gauss_data). 
        
        Parameters
        ----------
        ax, axis handle : if None creates a new figure
        
        Returns
        -------
        None
        
        See Also
        --------
        plot_gauss
        """
        # interpolate and cut off data
        
        newazim, newelev, newdata = self.interp()
        best_vals, covar = self.gauss_data()
        
        xn, yn = np.meshgrid(newazim, newelev)
        z = (xn,yn)
        
        data_fitted = self.gaussian_2d(z, *best_vals)
        
        # get metadata
        if self.var == 'SignalH':
            units = 'mW'
        else:
            metadata = generate_polvar_metadata(self.var)
            units = metadata['units']
        
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.hold(True)
            c = ax.imshow(newdata, cmap=plt.cm.jet, origin='bottom', 
                      extent=(xn.min(), xn.max(), yn.min(), yn.max()))
            CS = ax.contour(xn, yn, data_fitted.reshape(len(newelev), len(newazim)), 6, colors='w')         
            plt.clabel(CS, fontsize = 10, inline = 1)
            cbar = plt.colorbar(c)
            cbar.set_label(self.var + ' ' + units)
            plt.title('range' + str(self.range_r[self.gate]))
            plt.xlabel('Azimuth')
            plt.ylabel('Elevation')
            ax.ticklabel_format(useOffset=False)
        else:
            ax.hold(True)
            c = ax.imshow(newdata, cmap=plt.cm.jet, origin='bottom', 
                      extent=(xn.min(), xn.max(), yn.min(), yn.max()))
            box = ax.get_position()
            axc = plt.axes([box.x0*1.05 + box.width * 1.05, box.y0, 0.01, box.height])
            cbar = plt.colorbar(c, cax = axc)
            cbar.set_label(self.var + ' ' + units)
            CS = ax.contour(xn, yn, data_fitted.reshape(len(newelev), len(newazim)), 6, colors='w')
            plt.clabel(CS, fontsize = 10, inline = 1)
            

        
    def plot_scatter(self, ax = None):
        """
        Plots a scatter plot from the data
        
        Parameters
        ----------
        ax, axis handle : if None creates a new figure
        
        Returns
        -------
        None
        """
        
        ind, select = self.data_2d()
        azim = self.azimuth[ind]
        elev = self.elevation[ind]
        max_i = np.nanargmax(select)
        
        if self.var == 'SignalH':
            vmin = min(select)
            vmax = max(select)
            units = 'PowerH'
        elif self.var == 'Zh':
            units = 'dBZ'
            vmin = -10.
            vmax = 40.
        else:
            metadata = generate_polvar_metadata(self.var)
            vmin = metadata['valid_min']
            vmax = metadata['valid_max']
            units = metadata['units']
            
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.scatter(azim,elev,c=select, marker = 's', cmap = cm.jet, 
                       vmin = vmin, vmax =vmax)
            cbar = plt.colorbar()
            cbar.set_label(self.var + ' ' + units)
            plt.title('range' + str(self.range_r[self.gate]))
            plt.xlabel('Azimuth')
            plt.ylabel('Elevation')
            ax.ticklabel_format(useOffset=False)
        else:
            c = ax.scatter(azim,elev,c=select, marker = 's', cmap = cm.jet,
                       vmin = vmin, vmax =vmax)
            box = ax.get_position()
            axc = plt.axes([box.x0*1.05 + box.width * 1.05, box.y0, 0.01, box.height])
            cbar = plt.colorbar(c, cax = axc)
            cbar.set_label(self.var + ' ' + units)
            ax.plot([azim[max_i], azim[max_i]],[min(elev), max(elev)],'-k')
            ax.plot([min(azim), max(azim)],[elev[max_i], elev[max_i]], '-k')
        
    
    def plot_elev(self, ax = None):
        """
        Plots data from the elevation transect with a gaussian fitted curve
        
        Parameters
        ----------
        ax, axis handle : if None creates a new figure 
        
        Returns
        -------
        None
        """
        ind, select = self.data_elev()
        best_vals, covar = self.gauss_elev()
        x = self.elevation[ind]
        
        x_r = np.arange(x[0],x[-1],0.05)
        y_r = self.gaussian_1d(x_r, best_vals[0], best_vals[1], best_vals[2])
        
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.scatter(x,select,marker = 'o', label = 'observations')
            ax.plot(x_r, y_r, '-r', label = 'gauss_fit')
            ax.legend()
            plt.xlabel('Elevation')
            plt.ylabel(self.var)
        else:
            ax.scatter(select,x,marker = 'o', label = 'observations')
            ax.plot(y_r,x_r, '-r', label = 'gauss_fit')
            plt.xlabel(self.var)
            plt.ylabel('Elevation')
            plt.gca().invert_xaxis()       
        
        
    
    def plot_azim(self, ax = None):
        """
        Plots data from the azimuth transect with a gaussian fitted curve
        
        Parameters
        ----------
        ax, axis handle : if None creates a new figure
        
        Returns
        -------
        None
        """
        ind, select = self.data_azim()
        best_vals, covar = self.gauss_azim()
        x = self.azimuth[ind][::-1]
        select = select[::-1]
        
        x_r = np.arange(x[0],x[-1],0.05)
        y_r = self.gaussian_1d(x_r, best_vals[0], best_vals[1], best_vals[2])
        
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.scatter(x,select,marker = 'o', label = 'observations')
            ax.plot(x_r, y_r, '-r', label = 'gauss_fit')
            ax.legend()
            plt.xlabel('Azimuth')
            plt.ylabel(self.var)
        else:
             ax.scatter(x,select,marker = 'o', label = 'observations')
             ax.plot(x_r, y_r, '-r', label = 'gauss_fit')
             plt.xlabel('Azimuth')
             plt.ylabel(self.var)     
             plt.gca().invert_yaxis()
             plt.axis('tight')
        
    # private:
        
    ## -----------------------Fitting functions-------------------------- ##
    ## ------------------------------------------------------------------ ##
        
    def gauss_data_rad(self):
        """
        Fits data with 2D gaussian function and finds the best values and the
        covariance matrix. Fit is also stored in self.info_fit.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        best_vals, numpy array : best amp, cen and wid values for the gaussian
            function
            
        covar, numpy array : the estimated covariance of the optimal values for
            the parameters best_vals; use perr = np.sqrt(np.diag(pcov)) to 
            compute one standard deviation errors on the parameters
            
        See Also
        --------
        gauss_data
        """
        # select data for gaussian fitting using a circle with radius = self.doa
        # eliminating nan values and requiring data points to have higher values
        # then self.cutdB
        
        ind, select = self.data_2drad()
        
        newazim = np.take(self.azimuth, ind)
        newelev = np.take(self.elevation, ind)
        
        z = (newazim, newelev)
        
        # determine initial values
        amplitude = np.nanmax(select)
        #offset = np.min(select)
        max_i = np.nanargmax(select)
        
        xo = newazim[max_i]
        yo = newelev[max_i]

        # fit data with 2d gaussian function

        init_vals = (amplitude,xo,yo,self.sigx,self.sigy,self.theta) 
        
        best_vals, covar = opt.curve_fit(self.gaussian_2d, z, select, p0=init_vals)
        perr = np.sqrt(np.diag(covar))
        
        """
        self.info_fit['data'] = {'values': {'amp': best_vals[0], 'cenx': best_vals[1], 'ceny':best_vals[2],
              'sigx': best_vals[3], 'sigy':best_vals[4], 'thet':best_vals[5], 'off':best_vals[6]},
              'errors': {'amp': perr[0], 'cenx':perr[1], 'ceny':perr[2], 'sigx':perr[3],'sigy':perr[4],
              'thet':perr[5],'off':perr[6]}}"""
              
        self.info_fit['data'] = {'values': {'amp': best_vals[0], 'cenx': best_vals[1], 'ceny':best_vals[2],
              'sigx': best_vals[3], 'sigy':best_vals[4], 'thet':best_vals[5]},
              'errors': {'amp': perr[0], 'cenx':perr[1], 'ceny':perr[2], 'sigx':perr[3],'sigy':perr[4],
              'thet':perr[5]}}
        
        return best_vals, covar
        
    
    def gauss_data(self):
        """
        Interpolates data to a regular grid first (self.interp) then fits data 
        with 2d gaussian function and finds the best values and the
        covariance matrix. Fit info is also stored in self.info_fit.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        best_vals, numpy array : best amp, cen and wid values for the gaussian
            function
            
        covar, numpy array : the estimated covariance of the optimal values for
            the parameters best_vals; use perr = np.sqrt(np.diag(pcov)) to 
            compute one standard deviation errors on the parameters
            
        See Also
        --------
        gauss_data_rad
        """
        # interpolate and cut-off data for selected part if the scan
        
        newazim, newelev, newdata = self.interp()
        
        xn, yn = np.meshgrid(newazim, newelev)
        z = (xn,yn)
        
        # determine initial values
        
        amplitude = np.nanmax(self.data[self.gate,:])
        #offset = np.nanmin(self.data[self.gate,:])
        max_i = np.nanargmax(self.data[self.gate,:])
        
        yo = newelev[np.abs(newelev - self.elevation[max_i]).argmin()]
        xo = newazim[np.abs(newazim - self.azimuth[max_i]).argmin()]
        
        # fit data with 2d gaussian function
        
        init_vals = (amplitude,xo,yo,self.sigx,self.sigy,self.theta)
        
            
        best_vals, covar = opt.curve_fit(self.gaussian_2d, z, newdata.ravel(), p0=init_vals)
        perr = np.sqrt(np.diag(covar))
        
        """
        self.info_fit['data'] = {'values': {'amp': best_vals[0], 'cenx': best_vals[1], 'ceny':best_vals[2],
              'sigx': best_vals[3], 'sigy':best_vals[4], 'thet':best_vals[5], 'off':best_vals[6]},
              'errors': {'amp': perr[0], 'cenx':perr[1], 'ceny':perr[2], 'sigx':perr[3],'sigy':perr[4],
              'thet':perr[5],'off':perr[6]}}"""
              
        self.info_fit['data'] = {'values': {'amp': best_vals[0], 'cenx': best_vals[1], 'ceny':best_vals[2],
              'sigx': best_vals[3], 'sigy':best_vals[4], 'thet':best_vals[5]},
              'errors': {'amp': perr[0], 'cenx':perr[1], 'ceny':perr[2], 'sigx':perr[3],'sigy':perr[4],
              'thet':perr[5]}}
        
        return best_vals, covar
     
    
    def gauss_elev(self):
        """
        Fits elevation data with 1d gaussian function and finds the best values 
        and covariance matrix. Fit info is also stored in self.info_fit.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        best_vals, numpy array : best amp, cen and wid values for the gaussian
            function
            
        covar, numpy array : the estimated covariance of the optimal values for
            the parameters best_vals; use perr = np.sqrt(np.diag(pcov)) to 
            compute one standard deviation errors on the parameters
        """
        # find and reorganise data for selected part of the scan
        
        ind, select = self.data_elev()
        x = self.elevation[ind][::-1]
        select = select[::-1]
        
        # determine initial values
        data_s = self.data[self.gate,:]
        
        # convert data to linear values
            
        max_i = np.nanargmax(data_s)
        amplitude = data_s[max_i] 
        elevmid = self.elevation[max_i]
        
        # fit data with id gaussian function
        init_vals = [amplitude, elevmid, 2] 
        best_vals, covar = curve_fit(self.gaussian_1d, x, select, p0=init_vals)
        perr = np.sqrt(np.diag(covar))
        
        self.info_fit['elev'] = {'values': {'amp': best_vals[0], 'cen':best_vals[1], 'mid':best_vals[2]},
            'errors': {'amp':perr[0], 'cen':perr[1], 'mid':perr[2]}}
        
        return best_vals, covar
    
    def gauss_azim(self):
        """
        Fits azimuth data with 1d gaussian function and finds the best values 
        and covariance matrix
        
        Parameters
        ----------
        None
        
        Returns
        -------
        best_vals, numpy array : best amp, cen and wid values for the gaussian
            function
            
        covar, numpy array : the estimated covariance of the optimal values for
            the parameters best_vals; use perr = np.sqrt(np.diag(pcov)) to 
            compute one standard deviation errors on the parameters
        """
        # find and reorganise data for selected part of the scan
        
        ind, select = self.data_azim()
        x = self.azimuth[ind][::-1]
        select = select[::-1]
        
        # determine initial values
        data_s = self.data[self.gate,:]
        max_i = np.nanargmax(data_s)
        amplitude = data_s[max_i]
        azimmid = self.azimuth[max_i]
        
        # fit data with id gaussian function
        init_vals = [amplitude, azimmid, 2] 
        best_vals, covar = curve_fit(self.gaussian_1d, x, select, p0=init_vals)
        perr = np.sqrt(np.diag(covar))
        
        self.info_fit['azim'] = {'values': {'amp': best_vals[0], 'cen':best_vals[1], 'mid':best_vals[2]},
            'errors': {'amp':perr[0], 'cen':perr[1], 'mid':perr[2]}}
        
        return best_vals, covar
    
    def gaussian_1d(self, x, amp, cen, wid):
        """
        1D gaussian function to which to fit the data
        """
        return amp * np.exp(-(x-cen)**2 /wid)
        
    def gaussian_2d(self, xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
        """
        2D gaussian function to which to fit the data
        """
        xo = float(xo)
        yo = float(yo) 
        x = xy[0]
        y = xy[1]
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        #g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
            #+ c*((y-yo)**2)))
        g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
            + c*((y-yo)**2)))
            
        return g.ravel()
        
    ## ------------------------get data functions------------------------ ##
    ## ------------------------------------------------------------------ ##
    
    def data_2d(self):
        """
        finds the optimum value and data around it in the both the azimuth and
        elevation direction, which can then serve as input for a 2d gaussian 
        fit
        
        Parameters
        ----------
        None
        
        Returns
        -------
        ind, list : indices of data around the optimum value
        
        select, list : data around the optimum value
        """
        # select data
        data_s = self.data[self.gate,:]
        
        # find location max value
        max_i = np.nanargmax(data_s)
        
        if self.azim is None:
            azimmid = self.azimuth[max_i]
        else: 
            azimmid = self.azim
        if self.elev is None:
            elevmid = self.elevation[max_i]
        else: 
            elevmid = self.elev
            
        azimmax = azimmid + self.doa
        azimmin = azimmid - self.doa
        elevmax = elevmid + self.doe
        elevmin = elevmid - self.doe
        
        # find indices and data
        
        select = []
        ind = []
        
        for i in range(0,len(data_s)):
            if self.azimuth[i] > azimmin and self.azimuth[i] < azimmax and self.elevation[i] > elevmin and self.elevation[i] < elevmax:
                if np.isnan(data_s[i]):
                    pass
                else:
                    ind.append(i)
                    select.append(data_s[i])
                
        return ind, select
        
    def data_2drad(self, polvar = None):
        """
        finds the optimum value and data around it according to circle of 
        radius self.doa and stops at minval < self.cutdB which can then 
        serve as input dor a 2d gaussian fit
        
        Parameters
        ----------
        polvar, str : if given, extracts data for polvar instead of for the
            polvar given in self.polvar
        
        Returns
        -------
        ind, list : indices of data around optimum value
        
        select, list : data around the optimum value
        
        """
        # select data
        if polvar is None:
            data_e = self.data[self.gate,:]
            data_s = self.data[self.gate,:]
            maxval = self.cutdB 
        else:
            data_e = self.data[self.gate,:]
            data_s = self.file_handle.variables[polvar][self.gate, :] 
            maxval = -9999.
            
                
        # find location max value
        max_i = np.nanargmax(data_e)
        
        if self.azim is None:
            azimmid = self.azimuth[max_i]
        else: 
            azimmid = self.azim
        if self.elev is None:
            elevmid = self.elevation[max_i]
        else: 
            elevmid = self.elev
            
        # find sphere indices for graphing
            
        self.azim_sphere, self.elev_sphere = self.calc_sphere_coord(azimmid, elevmid, self.doa)
            
        # find indices and data inside circle
        select = []
        ind = []
        
        for i in range(0,len(data_s)):
            x = self.azimuth[i]
            y = self.elevation[i]
            if np.sqrt((x - azimmid)**2 + (y-elevmid)**2) <= self.doa:
                if np.isnan(data_s[i]):
                    pass
                elif data_s[i] < maxval:
                    pass
                else:
                    ind.append(i)
                    select.append(data_s[i])
                    
        return ind, select
    
    def data_elev(self):
        """
        finds the optimum value and data around it in the elevation direction,
        which can then serve as input for a 1d gaussian fit
        
        Parameters
        ----------
        None
        
        Returns
        -------
        ind, list : indices of data around the optimum value in the elevation
            direction
        
        select, list : data around the optimum value in the elevation direction
        """
        # select data
        data_s = self.data[self.gate,:]
        
        # find location max value
        max_i = np.nanargmax(data_s)
        
        if self.azim is None:
            azimmid = self.azimuth[max_i]
        else:
            azimmid = self.azim
            
        azimmin = azimmid - 0.125
        azimmax = azimmid + 0.125
            
        #azimmin = azimmid - 0.2
        #azimmax = azimmid + 0.2
        
        if self.elev is None:
            elevmid = self.elevation[max_i]
        else:
            elevmid = self.elev
        elevmax = elevmid + self.doe
        elevmin = elevmid - self.doe
        
        # make sure min and max values remain within measured axes
        
        if azimmax > np.max(self.azimuth):
            azimmax = np.max(self.azimuth)
        if azimmin < np.min(self.azimuth):
            azimmin = np.min(self.azimuth)
            
        if elevmax > np.max(self.elevation):
            elevmax = np.max(self.elevation)
        if elevmin < np.min(self.elevation):
            elevmin = np.min(self.elevation)
        
        # find indices and data
        
        select = []
        ind = []
        
        for i in range(0,len(data_s)):
            if self.elevation[i] > elevmin and self.elevation[i] < elevmax and self.azimuth[i] > azimmin and self.azimuth[i] < azimmax:
                if np.isnan(data_s[i]):
                    pass
                else:
                    ind.append(i)
                    select.append(data_s[i])
                
        return ind, select
        
    
    def data_azim(self):
        """
        finds the optimum value and data around it in the azimuthal direction,
        which can then serve as input for a 1d gaussian fit
        
        Parameters
        ----------
        None
        
        Returns
        -------
        ind, list : indices of data around the optimum value in the azimuthal
            direction
        
        select, list : data around the optimum value in the azimuthal direction
        """
        # select data
        data_s = self.data[self.gate,:]
        
        # find location max value
        max_i = np.nanargmax(data_s)
        
        if self.azim is None:
            azimmid = self.azimuth[max_i]
        else:
            azimmid = self.azim
        if self.elev is None:
            elevmid = self.elevation[max_i]
        else:
            elevmid = self.elev
        azimmax = azimmid + self.doa
        azimmin = azimmid - self.doa
        
        # make sure min and max values remain within measured axes
        
        if azimmax > np.max(self.azimuth):
            azimmax = np.max(self.azimuth)
        if azimmin < np.min(self.azimuth):
            azimmin = np.min(self.azimuth)
        
        elevmin = elevmid - 0.015
        elevmax = elevmid + 0.015
        
        if elevmax > np.max(self.elevation):
            elevmax = np.max(self.elevation)
        if elevmin < np.min(self.elevation):
            elevmin = np.min(self.elevation)
        
        elevmin = elevmid - 0.15
        elevmax = elevmid + 0.15
        
        # find indices and data
        
        select = []
        ind = []

        for i in range(0,len(data_s)):
            if self.azimuth[i] > azimmin and self.azimuth[i] < azimmax and self.elevation[i] > elevmin and self.elevation[i] < elevmax: 
                if np.isnan(data_s[i]):
                    pass
                else:
                    ind.append(i)
                    select.append(data_s[i])
                
        return ind, select
                
    
    def load_data(self):
        """
        Loads data from the given sphere calibration file
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        self.file_handle=Dataset(self.filename,'r')
        
        self.azimuth = self.file_handle.variables['Azimuth'][:]
        self.elevation = self.file_handle.variables['Elevation'][:]
        self.range_r = self.file_handle.variables['Range'][:]
        self.data = self.file_handle.variables[self.var][:] 
       
    
    def find_gate(self):
        """
        Finds the gate with the highest maximum returned value within a range
        of gates based on the resolution and the distance between the radar and
        the sphere. The min_gate and max_gate are based on observations of the
        data.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        the indice of the gate within the right range distance (radar-sphere)
        and with the highest value for self.var
        """
        # determine max and min gates
    
        resolution = self.range_r[1] - self.range_r[0]
        
        if resolution > 70:
            self.resolution = 75
            min_gate = 16
            max_gate = 20
        elif resolution > 25:
            self.resolution = 30
            min_gate = 39
            max_gate = 44
        else:
            self.resolution = 15
            min_gate = 80
            max_gate = 90
            
        indices = []
        max_vals = []
        for g in range(min_gate, max_gate):
            indices.append(g)
            data_slice = self.data[g]
            max_val = np.nanmax(data_slice)
            max_vals.append(max_val)
            
        return indices[max_vals.index(max(max_vals))]
        
    ## ---------------------Interpolation functions---------------------- ##
    ## ------------------------------------------------------------------ ##$
        
    def interp3(self, azim, elev, data):
        """
        Interpolates data to a regular, spherical grid
        
        Parameters
        ----------
        azim, np.array : azimuth values of the data
        
        elev, np.array : elevation values of the data
        
        data, np.array : data values
        
        Returns
        -------
        azim_interp, np.array : azimuth values interpolated data
        
        elev_interp, np.array : elevation values interpolated data
        
        grid, np.array : azimuth x elevation matrix interpolated data values
        
        See Also
        --------
        interp
        """
        
        # interpolation parameters
        
        azim_interp = np.arange(min(azim), max(azim), self.res_azim_interp)
        elev_interp = np.arange(min(elev), max(elev), self.res_elev_interp)
        
        x,y = np.meshgrid(azim_interp, elev_interp)
        
        points = np.vstack( (azim, elev) ).T
            
        # select data
        data_s = self.data[self.gate,:]
        
        # convert data to linear
        
        if self.var == 'SignalH':
            pass
        else:
            data_s = Conversion().logToLin(data_s, self.var)
        
        # find location max value
        max_i = np.nanargmax(data_s)
        
        if self.azim is None:
            azimmid = self.azimuth[max_i]
        else: 
            azimmid = self.azim
        if self.elev is None:
            elevmid = self.elevation[max_i]
        else: 
            elevmid = self.elev
            
        # take out values which are not within circle
            
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                if np.sqrt((x[i,j] - azimmid)**2 + (y[i,j]-elevmid)**2) >= self.doa:
                    x[i,j] = np.nan
                    y[i,j] = np.nan
                    
        grid = griddata(points, data, (x,y), method = 'linear')
        
        if self.var == 'SignalH':
            pass
        else:
            grid = Conversion().linToLog(grid, self.var)
        
        return azim_interp, elev_interp, grid        
        
    def interp(self):
        """
        Prepares data for the 2D gaussian fit:
        
        Interpolates data to a regular grid and cuts off new grid according to
        the data_2d function. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        newazim, list : azimuth values interpolated data
        
        newelev, list : elevation values interpolated data
        
        newdata, list : interpolated data values
        
        See Also
        --------
        interp3
        """
        # interpolation parameters
        
        azim_interp = np.arange(min(self.azimuth), max(self.azimuth), self.res_azim_interp)
        elev_interp = np.arange(min(self.elevation), max(self.elevation),self.res_elev_interp)
                
        data_grid = np.zeros((len(elev_interp), len(azim_interp)))
        num_samples = np.zeros((len(elev_interp), len(azim_interp)))
        
        # convert data to linear values for interpolation
        
        if self.var == 'SignalH':
            data = self.data[self.gate]
        else:
            data = self.data[self.gate]
            data = Conversion().logToLin(data, self.var)
            
        # interpolate to grid
        
        for dp in range(0,len(data)):
            idx = ( np.abs(elev_interp - self.elevation[dp]).argmin(), 
                   np.abs(azim_interp - self.azimuth[dp]).argmin() )
            num_samples[idx]=num_samples[idx]+1 
            data_grid[idx]=data_grid[idx]+data[dp]
                
        num_samples[num_samples==0]=float('nan')                         
        data_grid=data_grid/num_samples
        
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
        
        if self.var == 'SignalH':
            pass
        else:
            data_grid = Conversion().linToLog(data_grid, self.var)
            
        # cut off grid to focus on sphere
            
        ind, select = self.data_2d()
        
        elev_select = self.elevation[ind]
        azim_select = self.azimuth[ind]
        
        e_start = np.abs(elev_interp - (elev_select[0]-0.75)).argmin()
        e_end = np.abs(elev_interp - (elev_select[-1]+1.5)).argmin() + 1
        
        a_start = np.abs(azim_interp - (np.nanmin(azim_select)-0.3)).argmin()
        a_end = np.abs(azim_interp - (np.nanmax(azim_select)+1.25)).argmin()
        
        newazim = azim_interp[a_start:a_end]
        newelev = elev_interp[e_start:e_end]
        newdata = data_grid[e_start:e_end, a_start:a_end]
        
        """
        newdata = data_grid[40:81,40:81]
        newazim = azim_interp[40:81]
        newelev = elev_interp[40:81]"""
            
        return newazim, newelev, newdata
        
    ## -------------------------Other functions-------------------------- ##
    ## ------------------------------------------------------------------ ##
        
    def calc_sphere_coord(self, azimmid, elevmid, radius):
        """
        calculates the azimuth and elevation coordinates around the middle 
        of a sphere with a given radius
        
        Parameters
        ----------
        azimmid, float : azimuth value of the middle of the sphere
        
        elevmid, float : elevation value of the middle of the sphere
        
        radius, float : radius in degrees (azim or elev) of the sphere
        
        Returns
        -------
        azimcoord, list : list of azimuth coordinates of the sphere
        
        elevcoord, list : list of elevation coordinates of the sphere
        """
        azimcoord = []
        elevcoord = []
        for i in range(0,361):
            rad = i*np.pi/180
            azimcoord.append( azimmid + radius*math.sin(rad) )
            elevcoord.append( elevmid + radius*math.cos(rad) )
            
        return azimcoord, elevcoord        
       
        
    def create_fit_text(self, ttype):
        """
        Creates a textstring for plots containing information on the gaussian 
        fits
        
        Parameters
        ----------
        ttype, str : tells function whether to include 2D gaussian fit data (2d)
            or not (1d)
            
        Returns
        -------
        outstring, str : string containing information from self.info_fit
        """
        cutoff = '--------------------------------------------'
        sep = '\n' 
        msep = '   '
        #ssep = '      '
        tsep = "\t"
        header = msep + 'var' + tsep + tsep + 'est.val' + tsep + tsep + 'error'
        
        types = ['values', 'errors']
        
        if ttype == '2d':
            values = ['amp', 'cen','mid', 'cenx', 'ceny','sigx','sigy','thet', 'off']
            mykeys = ['azim','elev', 'data']
        else:
            values = ['amp', 'cen','mid']
            mykeys = ['azim','elev']
            
        measval = ( ('measured max val: %.3f') %(np.nanmax(self.data[self.gate])) )
        outstring = measval + sep + cutoff + sep + header + sep + cutoff + sep
        
        
        for key in mykeys:
            k_str = key + ':  ' + sep
            for v in values:
                #corr = len(ssep) + (5 - len(('%.3f')%(self.info_fit[key][types[0]][v])) )
                #csep = ' '*corr
                try:
                    v_str = ( (msep + v + tsep + tsep + '%.3f' + tsep + tsep + '%.3f' + sep)%
                    ( self.info_fit[key][types[0]][v], self.info_fit[key][types[1]][v]))
                    k_str += v_str
                except KeyError:
                    pass
                
            
            outstring += k_str
            
        return outstring.expandtabs()
                
        
            
            
