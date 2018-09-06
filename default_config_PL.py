# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:36:40 2016

@author: fvanden

Configuration file for Profile Lab toolkit

"""

MCH_elev=[-0.2,0.4,1,1.6,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,11,13,16,20,25,30,35,40]
NYQUIST_VEL=[8.3, 9.6,8.3,12.4,11.0,12.4,13.8,12.4,13.8,16.5,16.5,16.5,20.6,20.6,20.6,20.6,20.6,20.6,20.6,20.6]


	# pathnames used within the ProfileLab environment
	# can be set and changed here

MY_PATHS = {
    'temp_files' : '/home/fvanden/Documents/EPFL/MATLAB/Data/Payerne_2014/Parsivel/tmpfiles',
    'MXPOL' : '/home/fvanden/Documents/thuis/PythonData/MXPOL',
    'DX50' : '/home/fvanden/Documents/EPFL/Data',
    'ALB' : '/media/fvanden/Elements/EPFL/thuis/PythonData/MSRadarEx',
    'DOL': '/media/fvanden/Elements/EPFL/thuis/PythonData/MSRadarEx',
    'PPM' : '/media/fvanden/Elements/EPFL/thuis/PythonData/MSRadarEx',
    'MLE' : '/media/fvanden/Elements/EPFL/thuis/PythonData/MSRadarEx',
    'met_files' : '/home/fvanden/Documents/EPFL/MATLAB/Data/Payerne_2014',
    'parsivel_files' : '/media/fvanden/Elements/EPFL/MATLAB/Data/Payerne_2014/Parsivel/testfiles',
    
    # output files
    'RHI_out' : '/home/fvanden/Documents/EPFL/Python/ProfileLab_output/RHI/',
    'PPI_out' : '/home/fvanden/Documents/EPFL/Python/ProfileLab_output/PPI/',
    'CFAD_out': '/home/fvanden/Documents/EPFL/Python/ProfileLab_output/CFAD/' , 
    'VP_out' : '/home/fvanden/Documents/EPFL/Python/ProfileLab_output/VP/',
    'RR_tab_out' : '/home/fvanden/Documents/EPFL/Python/ProfileLab_output/RR_tab/',
    'VP_files' : '/home/fvanden/Documents/EPFL/Python/ProfileLab_output/VP_files'
    }
    
    # radar information 
 
RADAR_INFO = {
    'coordinates' : {
        'ALB' : [47.284,8.512],
        'DOL' : [46.425,6.099],
        'PPM' : [46.371,7.487],
        'MLE' : [46.041,8.833],
        'DX50' : [46.8425,6.9184],
        'MXPOL' : [46.8133,6.9428]
        },
    'altitude': {
        'ALB' : 938,
        'DOL' : 1682,
        'PPM' : 2937,
        'MLE' : 1626,
        'DX50': 451,
        'MXPOL': 489
        },
    'searchkey' : {
        'ALB' : 'PHA*hdf*',
        'DOL' : 'PHD*hdf*',
        'PPM' : 'PHP*hdf*',
        'MLE' : 'PHL*hdf*',
        'DX50' : None,
        'MXPOL' : None        
        },
    'radarID' : {
        'ALB' : 'ALB',
	'A':'ALB',
        'DOL' : 'DOL',
	'D':'DOL',
        'PPM' : 'PPM',
	'P':'PPM',
        'MLE' : 'MLE',
	'M':'MLE',
        'DX50' : 'DX50',
        'MXPOL' : 'MXPOL'
        },
    'dbbeam' : {
        'ALB' : 1.,
        'DOL' : 1.,
        'PPM' : 1.,
        'MLE' : 1.,
        'MXPOL' : 1.4,
        'DX50' : 1.27
        },
     'elevations': {
        'ALB': MCH_elev,
        'DOL': MCH_elev,
        'PPM': MCH_elev,
        'MLE' : MCH_elev,
        'DX50': None,
        'MXPOL': None
        }

    }
    
PARSIVEL_INFO = {
    'coordinates' : {
        '10' : [46.8872,7.0141],
        '20' : [46.9783,7.1300],
        '30' : [46.8425,6.9184],
        '40' : [46.8133,6.9428],
        '50' : [46.8115,6.9424],
        '2DVD' : [46.8115,6.9424]
        },
    'altitude' : {
        '10' : 435,
        '20' : 433,
        '30' : 451,
        '40' : 489,
        '50' : 489,
        '2DVD' : 489,
        },
    'parsID' : {
        '10' : '10',
        '20' : '20',
        '30' : '30',
        '40' : '40',
        '50' : '50',
        '2DVD' : '2DVD'
        },
    'long_name' : {
        '10' : 'HARAS Avanches',
        '20' : 'Bellechasse Airport',
        '30' : 'Payerne Airport',
        '40' : 'Payerne MCH roof',
        '50' : 'PAY station',
        '2DVD' : '2D Video Disdrometer'
        }

}

MY_METADATA = {

	'nyq_vel' : NYQUIST_VEL,

        # Metadata for instrument tables

        'Radar_info' : {
            'searchkey' : None,
            'coordinates' : None,
            'altitude' : None, 
            'dbbeam' : None,
            'filepath' : None,
            'radarID' : None},
            
        'Parsivel_info' : {
            'parsID' : None, 
            'coordinates' : None,
            'altitude' : None,
            'filepath' : None,
            'long_name': None},
    
        # Metadata for VerticalProfile attributes
        
        'Time' : {
            'units': 'Python time struct',
            'standard_name': 'time',
            'long_name': 'time of measurement'},                    
        
        'Profile_Type' : {
            'standard_name' : 'Profile type',
            'long_name' : 'time_average, spatial_average or instantaneous profile'},
            
        'Polvar' : {
            'units' : None,
            'standard_name' : None,
            'short_name' : None,
            'long_name' : None,
            'valid_min': None, 
            'valid_max': None,
            'plot_interval' : None},
            
        'Time_average_over' : {
            'units' : 'minutes',
            'standard_name' : 'time_average_over',
            'long_name' : 'time period over which average vertical profile has been calculated'},
            
        'Spatial_average_over' : {
            'units' : 'metres',
            'standard_name' : 'spatial_average_over',
            'long_name' : 'spatial extent over which average vertical profile has been calculated'},
            
        'Scan_list' : {
            'standard_name' : 'scan_list',
            'long_name' : 'list of databasenames for scans used to create vertical profile'},
            
        'Vp_calculation' : {
            'standard_name' : 'vp_calculation',
            'long_name' : 'dictionary with information on how vp calculation was performed'},
            
        'Profiles' : {
            'standard_name' : 'profiles',
            'long_name' : 'extracted vertical profiles',
            'cartesian_locations_on_grid': None,
            'cartesian_locations_distance_from_radar' : None,
            'latitude_locations' : None,
            'longitude_locations' : None,
            'type' : None},
            
        'Grid_info' : {
            'standard_name' : 'grid_info',
            'long_name' : 'information concerning grid projection on from which profiles were extracted',
            'latcoords' : None,
            'loncoords' : None },
            
        'Profile_coords_lat' : {
            'long_name': 'Latitude coordinates of vertical profile',
            'units': 'degrees_north',
            'standard_name': 'latitude',
            'valid_min': -90.,
            'valid_max': 90.},
            
        'Profile_coords_lon' : {
            'long_name': 'Longitude coordinates of vertical profile',
            'units': 'degrees_east',
            'standard_name': 'longitude',
            'valid_min':-180.,
            'valid_max':180.},
            
        # metadata for grid
            
        'projection' : {
            'standard_name' : 'projection',
            'long_name' : 'information on the algorithm used for the projection of the instrument',
            'proj': None,
            'script': None,
            'date_script': None },
            
        # Metadata for MetInfo attributes
            
        'timeperiod' : {
            'units' : 'Python time struct',
            'standard_name' : 'timeperiod',
            'long_name' : 'time periods for start/end of measurement'},
            
        'location' : {
            'standard_name' : 'location',
            'long_name' : 'geographical location and name of the meteorological station of the meteorological data',
            'ID' : None, 
            'station': None,
            'canton' : None,            
            'lat' : None,
            'lon' : None,
            'altitude' : None,
            'Ind_OMM' : None,
            'Ind_Nat' : None,
            'distance' : None,
            'variables' : None },
            
        'met_var' : {
            'standard_name' : None,
            'long_name' : None, 
            'units' : None,
            'valid_min' : None,
            'valid_max' : None },
            
        # Metadata for extra field names     
            
        'radar_echo_id': {
            'units': '-',
            'standard_name': 'radar_echo_id',
            'long_name': 'Radar Echo Identification',
            'coordinates': 'elevation azimuth range'},
            
            }
            
MY_POLARNAMES = {

        # Metadata for polarimetric short and long names

    'Zh' : ['Reflectivity','Reflectivity','dBZ', 0., 55.,1.],
    'Zhc' : ['Corr refl', 'Corrected_ReflectivityH','dBZ', 0., 55., 1.],
    'Zdr' : ['Diff. reflectivity','Differential reflectivity', 'dB', -1., 5.,0.1],
    'Zdrc': ['Corr. Diff. reflectivity','Corrected_Differential_Reflectivity','dB',-1., 5.,0.1],
    'Kdp' : ['Spec. diff. phase','Specific differential phase','deg/km',-2., 7., 0.1],
    'Kdpc' : ['Corr. spec. diff. phase','Corrected_SpecificDifferentialPhaseShift','deg/km',-2., 7., 0.1],
    'Phidp' : ['Diff. phase','Differential phase', 'deg',0., 150.,1.],
    'Phidpc' : ['Corr. diff. phase','Corrected_DifferentialPhaseShift', 'deg',0., 150.,1.],
    'Rhohv' : ['Copolar corr. coeff','Copolar correlation coefficient', '-',0.57, 1., 0.05],
    'Rhohvc' : ['Corr. copolar corr. coeff','Corrected_CopolarCorrelation', '-',0.57, 1., 0.05],
    'ZhCorr' : ['Att. corr reflectivity','Attenuation corrected reflectivity', 'dBZ', 0., 55.,1.], 
    'ZdrCorr' : ['Att corr. diff. reflectivity.','Attenuation corrected reflectivity','dB', 0., 3., 0.1],
    'RVel' : ['Mean doppler velocity','Mean doppler velocity','m/s', -15., 15.,0.5],
    'Sw' : ['Spectral Width','Spectral Width','m2/s2', 0., 3., 0.1],
    'Zv' : ['Vert. reflectivity', 'Vertical reflectivity','dBZ', 0., 45., 1.],
    'Clut' : ['Clutter', 'Output clutter algorithm','-',0.,100.,10.], 
    'TEMP' : ['Temperature', 'Cosmo temperature field','Celcius',-60,60,1.],
    'TYPECLUS2' : ['Hydroclass', 'Hydrometeor_type_from_Besic1','-',0.,200.,25.],
    'PROB' : ['Entropy', 'Hydrometeor_type_probability','-',0.,200.,25.],
    'corrected_Z' : ['corrected_Z', 'Clutter filtered reflectivity', 'dBZ', 0., 55., 1.]        

    }
    
MY_METKEYS = {
    
        # Metadata IDAWEB MeteoSwiss standard names
    
    'rre150z0' : ['Precipitation','Precipitation, ten minutes total', 'mm', 0., 800.],
    'precrate' : ['Precipitation_rate', 'Precipitation rate, based on ten minutes total','mm/h',0.,1800.],
    'prec_pars': ['Precipitation_rate', 'Precipitation rate, based on 30 min measurement','mm/h', 0.,1800.],
    'fu3010z1' : ['Windgust','Gust peak (one second), maximum', 'km/h', 0., 234.],
    'fu3010z0' : ['Windspeed', 'Wind speed, ten minutes mean','km/h', 0., 234.],
    'dkl010za' : ['Winddirection','Wind direction, standard deviation','degrees',0., 360.],
    'tre200s0' : ['Temperature', 'Air temperature 2 m above ground, current value','degrees celcius',-50., 50.],
    'wat000sw' : ['SYNOP','SYNOP: present weather at the time \n of observation (WMO 4677)','Code',None, None],
    'pp0qffs0' : ['Pressure','Pressure reduced to sea level (QFF), current value','hPa',600., 1100],
    'wkwtg3d0' : ['GWT_500_hPa','GWT with 26 classes based on 500 hPa geopotential \n (3E-20E,41N-52N)','Code', None, None],
    'wkwtp3d0' : ['GWT_surface', 'GWT with 26 classes based on surface pressure \n (3E-20E,41N-52N)', 'Code', None, None]
    
    }

FIGURE_PRESETS = {

        # figure presets for labplots (ProfileLab.Plotting.labplots)

    'default_figure':{
        'figsize': [18, 6],
        'height_lim': 6,
        'colormap' : 'jet',
        'out_type' : '.png'
        },

    'radar_echo_id' : {
        'figsize' : [18, 6],
        'height_lim' : 6,
        'colormap' : ['#0066CC', '#66FF66','#66FFFF','#FFFF66','#FF9933', '#990000'],
        #'bounds' : [0,11,21,31,41,51,61],
        'bounds' : [5,15,25,35,45,55,65],
        #'bounds_values' : [5,15,25,35,45,55,65],
        'bounds_values' : [10,20,30,40,50,60,70],
        'tick_labels' : ['Precip','Z','V', 'ZDR', 'RHO', 'Kdp'],     
        
        },
        
    'CFAD' : {
        'ZDR' : [[-3.0, 3.5, 0.6],[-2.7,3.2,0.6]],
        'Zdr' : [[-3.0, 3.5, 0.6],[-2.7,3.2,0.6]],
        'Z' : [[0.,56.,5.],[2.5,55.,5.]],
        'Kdp' : [[-4.,4.1,0.7],[-3.65,4.,0.7]],
        'RHO' : [[0.6, 1.1,0.05],[0.625,1.1,0.05]]
        }

}

VP_PRESETS = {

        # default presets for vertical profile extraction
        
        'rangemin': 500,
        'interval': None,
        'rangemax': 5000,
        'norm': 'off'
        }

PREPROCESSING_ORDER = {
    
        # list of order for preprocessing scheme
    
        'ATT' : 1,
        'CLUT': 2,
} 
#    
