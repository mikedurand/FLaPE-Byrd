'''
    PullGageData - functions to pull data from various agencies. so far this is just USGS.
    
'''

import dataretrieval.nwis as nwis # for pulling USGS data
import requests # for pulling SWOT data via Hydrochron
import numpy as np
import pandas as pd
import json
from io import StringIO #for parsing the csv


def PullUSGS(usgsid,logoutput):
    # example usage from https://www.hydroshare.org/resource/c97c32ecf59b4dff90ef013030c54264/
    parameterCode = "00060" # Discharge
    startDate = "2023-01-01"
    endDate = "2025-02-21"

    dfinfo = nwis.get_record(sites=usgsid, service='site')

    Qiv = nwis.get_iv(sites=usgsid, parameterCd=parameterCode, start=startDate, end=endDate) 
    
    nretrv=len(Qiv[0])

    print('  Retrieved ', nretrv,  ' instantaneous USGS data values.',file=logoutput)    
    
    if nretrv==0:
        return None,nretrv
    
    # stash data in dictionary
    QUSGS={
    'siteNumber': usgsid,
    'siteinfo': dfinfo,
    'Qiv' : Qiv[0],
    'NODATA' : -999999.0
    #'pcode' : parameterCode
    }
    
    #determine parameter code    
    if parameterCode in list(QUSGS['Qiv'].columns):
        QUSGS['pcode']=parameterCode
    else:
        for col in list(QUSGS['Qiv'].columns):
            if col[-3:] != '_cd':
                if col[0:5] == parameterCode:
                    QUSGS['pcode']=col       
    
    # set all nodata values to np nans
    # QUSGS['Qiv'].loc[QUSGS['Qiv'][parameterCode]==QUSGS['NODATA'],parameterCode]=np.nan
    QUSGS['Qiv'].loc[QUSGS['Qiv'][QUSGS['pcode']]==QUSGS['NODATA'],QUSGS['pcode']]=np.nan
    
    return QUSGS,nretrv

# def PullSWOT(reachid,logoutput):
#     # define command to get SWOT data
#     baseurl='https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?'

#     time='&start_time=2023-01-01T00:00:00Z&end_time=2025-02-21T00:00:00Z&'

#     fields=['cycle_id','pass_id','time_str','wse','width','slope2','slope','reach_q','p_width',
#             'p_length','xtrk_dist','ice_clim_f','reach_q_b','dark_frac','obs_frac_n',
#             'xovr_cal_q']

#     fieldstr='&fields='
#     for field in fields:
#         fieldstr+=field
#         if not(field==fields[-1]):
#             fieldstr+=','

#     # define API URL
#     url=baseurl + 'feature=Reach&feature_id=' + reachid + time + 'output=csv' + fieldstr
#     # pull data from HydroChron into res variable
#     res = requests.get(url)
    
#     # load data into a dictionary
#     data=json.loads(res.text)

#     # check status
#     if 'status' in data and data['status']=='200 OK':
#         print('  Successfully pulled SWOT data and put in dictionary',file=logoutput)
#     else:
#         print('  Something went wrong with Hydrochron call: SWOT data not pulled or not stashed in dictionary correctly',
#               file=logoutput)
#         print('  data=',data,file=logoutput)  
    
#     # load into a dataframe
#     df=pd.read_csv(StringIO(data['results']['csv']))
    
#     # store everything into a dictionary
#     SWOT={
#         'reachid':reachid,
#         'df':df,
#         'NODATA':-999999999999.0
#     }
    
#     # set all nodata values to np nans
#     for field in fields:
#         SWOT['df'].loc[SWOT['df'][field]==SWOT['NODATA'],field]=np.nan
#     # for times, set to None instead of 'no_data'
#     SWOT['df'].loc[SWOT['df']['time_str']=='no_data','time_str']=None
    
#     return SWOT
    