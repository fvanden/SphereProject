#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################
import os
import imp
from conversion import Conversion
#################

"""
Load ProfileLab configuration from a config file.

Created on Thu Mar 17 12:20:55 2016

@author: fvanden

Revision history:   17.03.2016 - Created

                    
"""

# The path to the default configuration file
_dirname = os.path.dirname(__file__)
_DEFAULT_CONFIG_FILE = os.path.join(_dirname, 'default_config_PL.py')
#_DEFAULT_CONFIG_FILE = os.path.join(_dirname, 'config_PL_Valais.py')

## -------------------------------------------------------------------------- ##

def load_myconfig(filename = None):
    """
    Load ProfileLab configuration from a config file.
    
    Parameters
    ----------
    filename: str
        Filename of the configuration file. If None the default configuration
        file is loaded from the ProfileLab directory.
    """
    
    if filename is None:
        filename = _DEFAULT_CONFIG_FILE
        
    # private:
    
    global cfile
    global _DEFAULT_METADATA
    global _DEFAULT_POLARNAMES
    global _DEFAULT_METNAMES
    global _DEFAULT_RADAR_INFO
    global _DEFAULT_PARSIVEL_INFO
    global _DEFAULT_PATHS
    global _DEFAULT_FIGURE_PRESETS
    global _PREPROCESSING_ORDER
    
    cfile = imp.load_source('metadata_config', filename)
    _DEFAULT_METADATA = cfile.MY_METADATA
    _DEFAULT_POLARNAMES = cfile.MY_POLARNAMES
    _DEFAULT_METNAMES = cfile.MY_METKEYS
    _DEFAULT_RADAR_INFO = cfile.RADAR_INFO
    _DEFAULT_PARSIVEL_INFO = cfile.PARSIVEL_INFO
    _DEFAULT_PATHS = cfile.MY_PATHS
    _DEFAULT_FIGURE_PRESETS = cfile.FIGURE_PRESETS
    _PREPROCESSING_ORDER = cfile.PREPROCESSING_ORDER
    return _DEFAULT_METADATA
    
def get_dict(d, filename = None):
    """
    Return any of the configuration file dictionaries
    
    Parameters
    ----------
    d, string : name of the dictionary to be returned
    
    filename : name of the configuration file, if None the default configuration
    file will be used
    
    Returns
    -------
    mydict, dict : the requested dictionary
    
    """
    load_myconfig(filename = filename)
    
    try:
        mydict = getattr(cfile, d)
    except AttributeError:
        print( ('%s does not exist in %s')%(d, filename) )
     
    return mydict
    
def get_mymetadata(p, filename = None):
    """
    Return a dictionary of metadata for a given parameter, p.

    An empty dictionary will be returned in no metadata dictionary exists for
    parameter p.
    """
    load_myconfig(filename = filename)    
    
    if p in _DEFAULT_METADATA:
        return _DEFAULT_METADATA[p].copy()
    else:
        return {}
        
def generate_radar_table(radarname, filename = None):
    """
    Generates a table with basic radar info, based on the given (or default)
    configfile
    
    Parameters
    ----------
    radarname: str
        name of the radar (i.e. 'ALB' or 'A', 'MXPOL' etc)
    
    filename: str
        path and name of the configfile, if None, the default configfile is
        used
        
    Returns
    -------
    radar_table: dict
        table containing basic radar info
    """
    load_myconfig(filename = filename)
    
    if radarname in _DEFAULT_RADAR_INFO['radarID']:
        radarname = _DEFAULT_RADAR_INFO['radarID'][radarname]
        radar_table = get_mymetadata('Radar_info', filename = filename)
        for key in radar_table:
            if key in _DEFAULT_RADAR_INFO:
                radar_table[key] = _DEFAULT_RADAR_INFO[key][radarname]
            elif key == 'filepath':
                radar_table[key] = load_path(radarname, filename = filename)
            else:
                radar_table[key] = None
        return radar_table
    else:
        return None
        
def generate_parsivel_table(parsivel_id, filename = None):
    """
    """
    load_myconfig(filename = filename)
    
    if parsivel_id in _DEFAULT_PARSIVEL_INFO['parsID']:
        parsivel_table = get_mymetadata('Parsivel_info', filename = filename)
        for key in parsivel_table:
            if key in _DEFAULT_PARSIVEL_INFO:
                parsivel_table[key] = _DEFAULT_PARSIVEL_INFO[key][parsivel_id]
            elif key == 'filepath':
                parsivel_table[key] = load_path('parsivel_files', filename = filename)
            else:
                parsivel_table[key] = None
        return parsivel_table
    else:
        return None
                
        
def generate_polvar_metadata(polvar, filename = None):
    """
    """
    load_myconfig(filename = filename)   
    polvar = Conversion().findKey('MXPOL', polvar)
    
    if polvar in _DEFAULT_POLARNAMES:
        standard_name, long_name, units, valid_min, valid_max, plot_interval =  _DEFAULT_POLARNAMES[polvar]
    else:
        standard_name, long_name, units, valid_min, valid_max, plot_interval = None, None, None, None, None, None
        
    polvar_metadata = get_mymetadata('Polvar', filename)
    polvar_metadata['units'] = units
    polvar_metadata['standard_name'] = standard_name
    polvar_metadata['short_name'] = Conversion().findKey('MCH', polvar)
    polvar_metadata['long_name'] = long_name
    polvar_metadata['valid_min'] = valid_min
    polvar_metadata['valid_max'] = valid_max
    polvar_metadata['plot_interval'] = plot_interval
    
    
    return polvar_metadata
    
def generate_Met_metadata(idaname, filename = None):
    """
    """
    load_myconfig(filename = filename) 
    
    if idaname in _DEFAULT_METNAMES:
        standard_name, long_name, units, valid_min, valid_max =  _DEFAULT_METNAMES[idaname]
    else:
        standard_name, long_name, units, valid_min, valid_max = None, None, None, None, None
        
    met_metadata = get_mymetadata('met_var', filename)
    met_metadata['units'] = units
    met_metadata['standard_name'] = standard_name
    met_metadata['long_name'] = long_name
    met_metadata['valid_min'] = valid_min
    met_metadata['valid_max'] = valid_max
    
    return met_metadata
    
    
def load_path(pathname, filename = None):
    """
    """
    load_myconfig(filename = filename) 

    if pathname in _DEFAULT_PATHS:
        return _DEFAULT_PATHS[pathname]
    else:
        return {}
        
def get_figure_preset(figuretype, preset, filename = None):
    """
    """
    load_myconfig(filename = filename)
    
    if figuretype in _DEFAULT_FIGURE_PRESETS:
        if preset in _DEFAULT_FIGURE_PRESETS[figuretype]:
            return _DEFAULT_FIGURE_PRESETS[figuretype][preset]
        else:
            print( ('no %s for figure of type %s') %(preset, figuretype) )
    else:
        if preset in _DEFAULT_FIGURE_PRESETS['default_figure']:
            return _DEFAULT_FIGURE_PRESETS['default_figure'][preset]
        else:
            print( ('no %s for figure of type %s') %(preset, figuretype) )
    

    
