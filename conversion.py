#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################
import numpy as np
import itertools
#################

"""
Module containing class Conversion. Can also be executed as a main function.

Created on Mon Dec 21 14:41:11 2015

@author: fvanden

Revision history:   21.12.2015 - Created
                    03.01.2016 - digitalTodB function added
                    TO DO: add other conversions
                    27.01.2016 - log and lin transformations added
                                conversion no longer daughterclass of compute
                                createKeys and findKey added
                    08.02.2016 - createKeys extended so that Rhohv = RHO, V = Rvel
                                and Sw = W
                    12.08.2016 - Isztar's method added to refl2Rain
                    20.10.2017 - added gwt conversion

"""

## -------------------------------------------------------------------------- ##
class Conversion():

    """ 
    class Conversion: contains all kinds of conversions performed on data and 
    keys within the ProfileLab toolbox. i.e. Converts the diffent types of 
    polarimetric keys and reflectivity rainrate relations.
    
    Parameters
    ----------
    None : depends on the many functions
    
    Attributes
    ----------
    self.conventions : list of strings, the types of polarimetric conventions 
        we work with
    
    self.convertKeys : dict, lookup table for conversions between conventions
    
    """

    ## ------------------------------------------------------------------ ##
    ## Constructors/Destructors                                           ##
    ## ------------------------------------------------------------------ ##

    def __init__(self):
        """
        Initiates mother class Compute. As such, also inherits Data and 
        coordinate Grid.
        """
        self.conventions = ['MXPOL','MCH','LogLin']
        self.convertKeys = self.createKeys()
        
    def __del__(self):
        pass

    ## ------------------------------------------------------------------ ##
    ## Methods                                                            ##
    ## ------------------------------------------------------------------ ##

    # public:

    def Refl2Rain(self, reflectivity, method = 'Zh', A = 200, b = 5/8):
        """
        """
        if method == 'Zh':
            rainrate = (self.logToLin(reflectivity, 'Z') / A)**(b)
            
        if method == 'Isztar':
            x = reflectivity
            logRR = -2.3 +0.17*x -5.1e-3*x**2 +9.8e-5*x**3 -6e-7*x**4
            dBR = logRR * 10
            rainrate = dBR
            rainrate = self.logToLin(rainrate, 'Z')
            
        return rainrate
        
    def Rain2Refl(self, rainrate, method = 'Zh', A = 200, b = 5/8):
        """
        """
        if method == 'Zh':
            reflectivity = self.linToLog(rainrate**(1/(b))*A, 'Z')
            
            
        return reflectivity
            
                
    def logToLin(self, data, polvar):
        """
        Transforms logarithmic data into linear data, with a different 
        transformation depending on type of polarimetric variable.
        
        Parameters
        ---------- 
        data : list or numpy array, polarimetric data to be converted
        
        polvar : str, name of the polarimetric variable type for the data
        
        Returns
        -------
        varLin : list or numpy array : data converted into linear values

        """
        polvar = self.findKey('LogLin', polvar)
        data = np.asarray(data)
        if(polvar in ['Ph','Pv','SNRh','SNRv','Zh','Zv','MZh','MZv']):
            varLin=10**(0.1*data)
        elif(polvar=='Sw'):
            varLin=data**2
        else:
            varLin = data
            
        return varLin
        
    def linToLog(self, data, polvar):
        """
        Transforms linear data to logarithmic data, with a different 
        transformation depending on type of polarimetric variable.
        
        Parameters
        ----------
        
        data : list or numpy array, polarimetric data to be converted
        
        polvar : str, name of the polarimetric variable type for the data
        
        Returns
        -------
        varLin : list or numpy array : data converted into linear values

        """
        polvar = self.findKey('LogLin', polvar)
        data = np.asarray(data)
        if(polvar in ['Ph','Pv','SNRh','SNRv','Zh','Zv','MZh','MZv']):
            varLog = 10*np.log10(data)
        elif(polvar=='Sw'):
            varLog = np.sqrt(data)
        else:
            varLog = data
            
        return varLog
        
                
    def digitalTodB(self, Data, polvar):
        """
        Digital number to dB conversion. Automatically converts between MXPOL
        and MCH conventions.
        
        Parameters
        ----------
        
        Data : dict, with instrumentnames in the keys and data values 
        
        polvar : str, polarimetric variable name.
        
        Returns
        -------
        
        Data, dict with instrumentnames in keys and converted data

        """
        for i in range(0,len(Data)):
            for key in Data:
                if 'AL' or 'LD' or 'PM' in key.split('_')[0]:
                    if polvar == 'Z':
                        data = Data[key]
                        data = np.asarray(data)
                        data = data.astype(float)
                        output=np.zeros(data.shape)
                        output[data!=0]=(data[data!=0])*0.5
                        output[data==0]=float('nan')
                        Data[key] = output                                    
            
                    elif polvar == 'ZDR':
                        pass
                    else:
                        pass 
                    
        return Data
        
    def findKey(self, convention, polvar):
        """
        Finds the correct variable name for a given convention (MXPOL, MCH) and 
        a given variable name which was spelled with a different case or 
        according to a different convention. For example, MXPOL convention uses 
        'Z' for the reflectivity variable, but if a user inserted 'Zh' this 
        function will convert it to 'Z'. 
        
        Parameters
        ----------
        convention : str, for now, MCH, MXPOL or LogLin (for logarithmic/linear
            conversions)
            
        polvar : str, key of polarimetric variable to be converted
        
        Returns
        -------
        mykey : str, polarimertric variable key as used within the ProfileLab
            toolbox context
            
        """
        for key, value in self.convertKeys[convention].items():
            mykey = polvar
            if polvar in value:
                mykey = key
                break
                
        return mykey
        
    def gwtConv(self, gwtcode):
        """
        Converts GWT26 code into flow direction and cyclonicity
        
        Parameters
        ----------
        gwtcode : int, gwt number from MeteoSwiss observations or
            2-tuple of int organised as: (flow direction (int), cyclonicity 
            (float)) 
        
        Returns
        -------
        output : the converted code
        
        See Also
        --------
        For GWT26 code:
        
        http://www.meteoschweiz.admin.ch/content/dam/meteoswiss/en/Ungebundene-Seiten/Publikationen/Fachberichte/doc/ab235.pdf
        
        The flow direction/cyclonicity code is of my own invention and is 
        organised as follows:
        
        directions are given by numbers 1 to 8, starting with 1 for North and 
        in clockwise direction ending with 8 for NorthWest. Cyclonicity is 
        given by a floating number: 1 for cyclonic, 0 for anticyclonic and 
        0.5 for indifferent.
        
        """
        
        conversion_table = {
            'nan' : 'nan',
            1 :  (7, 1.),
            2 :  (6,1.),
            3 :  (8,1.),
            4 :  (1,1.),
            5 :  (2,1.),
            6 :  (3, 1.),
            7 :  (4,1.),
            8 :  (5,1.),
            9 :  (7,0.),
            10 : (6,0.),
            11 : (8,0.),
            12 : (1,0.),
            13 : (2,0.),
            14 : (3,0.),
            15 : (4,0.),
            16 : (5,0.),
            17 : (7,.5),
            18 : (6,.5),
            19 : (8,.5),
            20 : (1,.5),
            21 : (2,.5),
            22 : (3,.5),
            23 : (4,.5),
            24 : (5,.5),
            25 : (0,1.),
            26 : (0,.5),
            }
                
        if isinstance(gwtcode, int):
            output = conversion_table[gwtcode]
        elif np.isnan(gwtcode):
            output = np.nan
        elif isinstance(gwtcode, tuple):    
            for key, value in conversion_table.items():
                if value == gwtcode:
                    output = key
        else:
            print("not a valid input, try integer or tuple")
            output = []
            
        return output
        
    # private:

    def createKeys(self):
        """
        Creates dictionary for key conversion
        """
        MCH = ['Z','ZV','ZDR','PHIDP','V', 'W', 'RHO', 'MPH', 'CLUT', 'STA1', 'STA2', 'WBN']
        MXPOL = ['Zh','Zdr', 'Phidp', 'RVel','Sw','Rhohv', 'Zv', 'Clut']
        LogLin = ['Ph','Pv','SNRh','SNRv','Zh','Zv','MZh','MZv','Sw']
        convertkeys = {}
        for conv in self.conventions:
            convertkeys[conv] = {}
            if conv == 'MXPOL':
                variables = MXPOL
            elif conv == 'MCH':
                variables = MCH
            elif conv == 'LogLin':
                variables = LogLin
            else:
                # add other conventions here
                pass
            for var in variables:
                convertkeys[conv][var] = []
                convertkeys[conv][var] = list((map(''.join, itertools.product(*zip(var.upper(), var.lower())))))
                if var == 'Zh':
                    convertkeys[conv][var].extend('z')
                    convertkeys[conv][var].extend('Z')
                elif var == 'Z':
                    s = 'Zh'
                    convertkeys[conv][var].extend(list((map(''.join, itertools.product(*zip(s.upper(), s.lower()))))))
                elif var == 'RHO' or var == 'Rhohv':
                    if conv == 'MCH':
                        s = 'Rhohv'
                    else:
                        s = 'RHO'
                    convertkeys[conv][var].extend(list((map(''.join, itertools.product(*zip(s.upper(), s.lower()))))))
                elif var == 'Sw' or var == 'W':
                    if conv == 'MCH':
                        s = 'Sw'
                    else:
                        s = 'W'
                    convertkeys[conv][var].extend(list((map(''.join, itertools.product(*zip(s.upper(), s.lower()))))))
                elif var == 'RVel' or var == 'V':
                    if conv == 'MCH':
                        s = 'RVel'
                    else:
                        s = 'V'
                    convertkeys[conv][var].extend(list((map(''.join, itertools.product(*zip(s.upper(), s.lower()))))))
                else:
                    pass
                            
        return convertkeys
                          

## -------------------------------------------------------------------------- ##


if __name__ == '__main__':
        test = Conversion()

