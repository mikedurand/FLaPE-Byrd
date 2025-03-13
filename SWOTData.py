'''
    SWOTData.py includes:
      - SWOTData class - filter SWOT data and matching to USGS
      - tukey_test function - do outlier detection
      - PullSWOT function - pull SWOT data from Hydrochron and write SWOT dataframe
      - ReadSWOT function - read an existing SWOT dataframe 
      
    by Mike, March 2025
'''

import numpy as np
import pandas as pd
from datetime import datetime
import requests 
import json
from io import StringIO 


class SWOTData:
    def __init__(self,logoutput,SWOTdict=None,SWOTfname=None):
        self.logoutput=logoutput
        if SWOTdict:
            print('  loading SWOT data from dictionary',file=self.logoutput)
            self.data=SWOTdict
        if SWOTfname:
            print('  loading SWOT+gage data from dataframe',file=self.logoutput)
            reachid=str(SWOTfname).split('/')[-1].split('_')[0]
            self.data={
                'reachid':reachid,
                'NODATA':-999999999999.0
            }       
            self.data['df']=pd.read_csv(SWOTfname)
            
    def threshold_filter(self,slope_min=None,slope2_min=None,dark_max=None):
        filters={
             'prior_width_min_m' : 80 ,
             'prior_slope_min' : 3.4e-5,
             'reach_length_min_m':7000,
             'cross_track_dist_min_m': 10000,
             'cross_track_dist_max_m': 60000,
             'ice_max' : 0,
             'target_bit_reach': 507510784, #FUNCTIONALLY EQUIVALENT TO _q<:2
             'dark_max' : 0.4,
             'obs_frac_min': 0.5,
             'xover_cal_q_max' : 1,
             'Tukey_number' : 1.5}
        
        if dark_max:
            filters['dark_max']=dark_max

        # apply threshold type filters
        self.data['df']['bad-thresh'] = \
            (    self.data['df']['p_width']        < filters['prior_width_min_m']) \
              | (self.data['df']['p_length']       < filters['reach_length_min_m']) \
              | (abs(self.data['df']['xtrk_dist']) < filters['cross_track_dist_min_m']) \
              | (abs(self.data['df']['xtrk_dist']) > filters['cross_track_dist_max_m']) \
              | (self.data['df']['ice_clim_f']       > filters['ice_max']) \
              | (self.data['df']['dark_frac']       > filters['dark_max']) \
              | (self.data['df']['obs_frac_n']       < filters['obs_frac_min']) \
              | (self.data['df']['xovr_cal_q']       > filters['xover_cal_q_max'])
        
        if slope_min:
            print('  applying slope minimum as threshold filter',file=self.logoutput)
            self.data['df']['bad-thresh']=( self.data['df']['bad-thresh']) \
                                      |   ( self.data['df']['slope'] < slope_min )
        
        print('  there are',sum(self.data['df']['bad-thresh']),'bad data points identified by the threshold swot data filters',file=self.logoutput)
    def bitwise_filter(self):
        targetbit=507510784
        self.data['df']['bad-bitwise']=np.bitwise_and(self.data['df']['reach_q_b'].astype(int),targetbit) > 0
        self.data['df']['bad-thresh-or-bit']=self.data['df']['bad-thresh'] | self.data['df']['bad-bitwise']
        
        print('  there are',sum(self.data['df']['bad-thresh-or-bit']),'bad data points identified by combined threshold & bitwise filters',file=self.logoutput)
        
    def outlier_test(self,Verbose=False):
        # test for outliers
        
        if all(self.data['df']['bad-thresh-or-bit']):            
            self.data['df']['tukey-outlier']=None
            self.data['df']['bad']=True # set all to true
            if Verbose:
                print('  there are',sum(self.data['df']['bad']),'bad data points identified by combined threshold, bitwise & outlier filters',file=self.logoutput)
            return        

        self.data['df']['tukey-outlier']=(tukey_test(self.data['df']['wse'],mask=self.data['df']['bad-thresh-or-bit'])) \
                            | (tukey_test(self.data['df']['width'],mask=self.data['df']['bad-thresh-or-bit'])) \
                            | (tukey_test(self.data['df']['slope'],mask=self.data['df']['bad-thresh-or-bit'])) 
                            # | (tukey_test(df['slope2'].astype(float),mask=df['bad-thresh'])) 
        self.data['df']['bad']=(self.data['df']['bad-thresh']) | (self.data['df']['bad-bitwise']) | (self.data['df']['tukey-outlier'])
        print('  there are',sum(self.data['df']['bad']),'bad data points identified by combined threshold, bitwise & outlier filters',file=self.logoutput)
        
    def slope_limit(self,p_slope,opt=0):
        
        if opt==0:
            # colin's suggestion was if if any slopes are less than zero, set all to the prior slope.
            Smin=3.4e-50 #colin's analog for small positive number

            Slim=3.4e-5

            if p_slope > Slim:
                Slim=p_slope
                print('  Using SWORD slope for this reach, S limit=',Slim,file=self.logoutput)
            else:
                print('  Using arbitrary slope for this reach, S limit=',Slim,file=self.logoutput)

            if any(self.data['df']['slope']<Smin):
                print('  setting all slope values to',Slim,file=self.logoutput)
                self.data['df'].loc[:,'slope']=Slim # setting to a reasonable low slope
            else:
                print('  SWOT slope measurements used',file=self.logoutput)
            if any(self.data['df']['slope2']<Smin):
                print('  setting all slope2 values to',Slim,file=self.logoutput)
                self.data['df'].loc[:,'slope2']=Slim # setting to a reasonable low slope
            else:
                print('  SWOT slope2 measurements used',file=self.logoutput)    
        else:
            slope_lim=opt 
            print('  setting all slope and slope2 values less than ',slope_lim,'to ',slope_lim,file=self.logoutput)
            self.data['df'].loc[self.data['df']['slope']<slope_lim,'slope']=slope_lim
            self.data['df'].loc[self.data['df']['slope2']<slope_lim,'slope2']=slope_lim
        
    def MatchUSGS(self,QUSGS):
        # pull out USGS times as timestamps
        tsdt=list(pd.to_datetime(QUSGS['Qiv'].index.array)) #datetime ts
        ts=np.array([t.timestamp() for t in tsdt])
        
        # pull out SWOT times as timestamps
        tswots=[]
        i=0
        for t in list(self.data['df']['time_str']):   
            if not t:
                tswots.append(np.nan)
            else:
                tswots.append(datetime.fromisoformat(t[:-1]).timestamp())     
                
        # interpolate and convert units
        self.data['df']['Qgage']=np.interp(tswots,ts,np.array(QUSGS['Qiv'][QUSGS['pcode']])) # interpolate gage data to swot times
        self.data['df']['Qgage']*=0.3048**3
        
        
def tukey_test(data, tukey_num=1.5,mask=None):
    
    # compute outlier gates based only on unmasked data
    if mask is None:
        p_low=np.percentile(data,25)
        p_hi=np.percentile(data,75)        
    else:
        p_low=np.percentile(data[~mask],25)
        p_hi=np.percentile(data[~mask],75)        

    
    iqr=p_hi-p_low
    
    #print('25th percentile:',p_low,'; 75th percentile:',p_hi,'; IQR=',iqr)
    
    # outlier detection performed on masked and unmasked data
    outlier=(data < p_low-tukey_num*iqr) | (data > p_hi+tukey_num*iqr)
    
    return outlier     

def PullSWOT(reachid,logoutput):
    # define command to get SWOT data
    baseurl='https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?'

    time='&start_time=2023-01-01T00:00:00Z&end_time=2025-02-21T00:00:00Z&'

    fields=['cycle_id','pass_id','time_str','wse','width','slope2','slope','reach_q','p_width',
            'p_length','xtrk_dist','ice_clim_f','reach_q_b','dark_frac','obs_frac_n',
            'xovr_cal_q']

    fieldstr='&fields='
    for field in fields:
        fieldstr+=field
        if not(field==fields[-1]):
            fieldstr+=','

    # define API URL
    url=baseurl + 'feature=Reach&feature_id=' + reachid + time + 'output=csv' + fieldstr
    # pull data from HydroChron into res variable
    res = requests.get(url)
    
    # load data into a dictionary
    data=json.loads(res.text)

    # check status
    if 'status' in data and data['status']=='200 OK':
        print('  Successfully pulled SWOT data and put in dictionary',file=logoutput)
    else:
        print('  Something went wrong with Hydrochron call: SWOT data not pulled or not stashed in dictionary correctly',
              file=logoutput)
        print('  data=',data,file=logoutput)  
    
    # load into a dataframe
    df=pd.read_csv(StringIO(data['results']['csv']))
    
    # store everything into a dictionary
    SWOT={
        'reachid':reachid,
        'df':df,
        'NODATA':-999999999999.0
    }
    
    # set all nodata values to np nans
    for field in fields:
        SWOT['df'].loc[SWOT['df'][field]==SWOT['NODATA'],field]=np.nan
    # for times, set to None instead of 'no_data'
    SWOT['df'].loc[SWOT['df']['time_str']=='no_data','time_str']=None
    
    return SWOT

def ReadSWOT(reachid,logfile,fname=None,reachid17=None):
    '''
         read in a SWOTData type dictionary, from a dataframe of multiple reaches, passes and cycles from ADT
         
         if we are doing a translation between SWORD16 and 17, then this will read data from SWORD17 files and return data which
         will then be saved into SWORD16 filenames.
         
         so it reads in reachid assuming this is the sword16 reachid. then it pulls the corresponding sword17 reachid, then reads
         that data, and passes back a dictionary tagged with the sword16 reachid
    
    '''
    
    if fname:
        df=pd.read_csv(fname)
        
        
    # # translation
    # if translationfile is not None:
    #     reachid17=int(translationfile[translationfile['v16_reach_id']==int(reachid)]['v17_reach_id'])
    #     print('  using translation file, mapping',reachid,'to sword17 reachid=',reachid17,file=logfile)
       
    if reachid17 is not None:
        df=df[df['reach_id']==np.int64(reachid17)]
    else:
        df=df[df['reach_id']==np.int64(reachid)]
    
    
    # note everything here down is identical to PullSWOT - should try to merge them
    fields=['cycle_id','pass_id','time_str','wse','width','slope2','slope','reach_q','p_width',
        'p_length','xtrk_dist','ice_clim_f','reach_q_b','dark_frac','obs_frac_n',
        'xovr_cal_q']
    
    # store everything into a dictionary
    SWOT={
        'reachid':reachid,
        'reachid17':reachid17,
        'df':df,
        'NODATA':-999999999999.0
    }
    
    # set all nodata values to np nans
    for field in fields:
        SWOT['df'].loc[SWOT['df'][field]==SWOT['NODATA'],field]=np.nan
    # for times, set to None instead of 'no_data'
    SWOT['df'].loc[SWOT['df']['time_str']=='no_data','time_str']=None

    
    return SWOT
