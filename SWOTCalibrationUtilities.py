'''
     SWOTCalibrationUtilities: helper functions for SWOTCalibration experiments
        - OutputReachCalibration - write out output dataframe
        - getresultdf - stack multiple result dataframes together. possibly obsolete
        - CalibrateSWOTReach - perform flow law calibration
        - SetOptions - set variables with the calibration experiment options
'''


import pandas as pd
import numpy as np
import os
# these assume this file is in same directory as other FLaPE-Byrd files
from RiverIO import RiverIO
from Domain import Domain
from ReachObservations import ReachObservations
from ReachTruth import ReachTruth
from FlowLawCalibration import FlowLawCalibration
from FlowLaws import MWAPN,AHGD,AHGW


# def ReadSWOT(reachid,logfile,fname=None,reachid17=None):
#     '''
#          read in a SWOTData type dictionary, from a dataframe of multiple reaches, passes and cycles from ADT
         
#          if we are doing a translation between SWORD16 and 17, then this will read data from SWORD17 files and return data which
#          will then be saved into SWORD16 filenames.
         
#          so it reads in reachid assuming this is the sword16 reachid. then it pulls the corresponding sword17 reachid, then reads
#          that data, and passes back a dictionary tagged with the sword16 reachid
    
#     '''
    
#     if fname:
#         df=pd.read_csv(fname)
        
        
#     # # translation
#     # if translationfile is not None:
#     #     reachid17=int(translationfile[translationfile['v16_reach_id']==int(reachid)]['v17_reach_id'])
#     #     print('  using translation file, mapping',reachid,'to sword17 reachid=',reachid17,file=logfile)
       
#     if reachid17 is not None:
#         df=df[df['reach_id']==np.int64(reachid17)]
#     else:
#         df=df[df['reach_id']==np.int64(reachid)]
    
    
#     # note everything here down is identical to PullSWOT - should try to merge them
#     fields=['cycle_id','pass_id','time_str','wse','width','slope2','slope','reach_q','p_width',
#         'p_length','xtrk_dist','ice_clim_f','reach_q_b','dark_frac','obs_frac_n',
#         'xovr_cal_q']
    
#     # store everything into a dictionary
#     SWOT={
#         'reachid':reachid,
#         'reachid17':reachid17,
#         'df':df,
#         'NODATA':-999999999999.0
#     }
    
#     # set all nodata values to np nans
#     for field in fields:
#         SWOT['df'].loc[SWOT['df'][field]==SWOT['NODATA'],field]=np.nan
#     # for times, set to None instead of 'no_data'
#     SWOT['df'].loc[SWOT['df']['time_str']=='no_data','time_str']=None

    
#     return SWOT

def OutputReachCalibration(Cal,rid):
    # designate names for flow law parameters
    param_names=['na','A0','x1'] # these change with flow law        
    
    n_params=len(Cal.param_est)
    
    if n_params<len(param_names):
        param_names=param_names[0:n_params]
    
    # designate names for error variables
    excludelist=['Qt','Qhat','D']
    errorvars=[var for var in   list(vars(Cal.Performance).keys()) if var not in  excludelist]    
    n_evs=len(errorvars)
    
    #combine params and error variables on to one array (eventually one df row)
    outnames=param_names+errorvars
    n_out=len(outnames)

    # initialize an array for the output data
    out_data=np.full( shape=(n_out,),fill_value=np.nan  )    

    # assign FLPs to the output array
    # out_data[0:3]=Cal.param_est
    out_data[0:n_params]=Cal.param_est
   
    # assign error stats to the output array 
    for i in range(n_evs):
        idx=n_params+i
        out_data[idx]=getattr(Cal.Performance,errorvars[i])
    
    # set up dataframe and write out
    dfout=pd.DataFrame(columns=outnames,data=out_data.reshape( (1,n_out)  ) ,index=[rid])
    
    return dfout

def getresultdf(idx,expdf,ExpDataDir):
    expsid=expdf.iloc[idx]['expsid']
    expid=expdf.iloc[idx]['expid']
    ExpsDir=ExpDataDir.joinpath(expsid)
    ExpDir=ExpsDir.joinpath(expid)
    ExpStatsDir=ExpDir.joinpath('fit+stats')
    # ExpStatsDir
    # get list of files
    calfiles=os.listdir(ExpStatsDir)
    calfiles=[file for file in calfiles if file[0]!='.']
    
    # assemble dataframe of all individual dfs
    dfs=[]
    for file in calfiles:
        dfs.append(pd.read_csv(ExpStatsDir.joinpath(file),index_col=0))

    resultdf=pd.concat(dfs)     
    
    return resultdf

def CalibrateSWOTReach(fname,logoutput,slope_opt='slope',area_opt='MetroMan',constrainhw=False,flowlawname='MWAPN',Verbose=False,lossfun='linear'):
    
    '''
        CalibrateReach uses FLaPE Byrd with a single flow law to calibrate a reach
            fname:  filename of SWOT hydrochron dataframe to read
            slope_opt: which slope data element to use: can be either slope or slope2
            
        
        this is only partly complete - e.g. getFlowLaw only has three flow laws coded so far
        
        by Mike, March 2025
    '''
    
    
    def getFlowLaw(flowlawname):
        if flowlawname=='MWAPN':
            FlowLaw=MWAPN(ReachDict['dA'],ReachDict['w'],ReachDict['S'],ReachDict['H'])
        elif flowlawname=='AHGD':
            FlowLaw=AHGD(ReachDict['dA'],ReachDict['w'],ReachDict['S'],ReachDict['H'])
        elif flowlawname=='AHGW':
            FlowLaw=AHGW(ReachDict['dA'],ReachDict['w'],ReachDict['S'],ReachDict['H'])
        else:
            # print('  unknown flowlaw option',file=logoutput)
            print('  unknown flowlaw option')
            return None
                  
        return FlowLaw
    
    
    # just runs a single flow law for now
    print('  running with ', slope_opt,file=logoutput)
    
    # read in SWOT+gage data and set up input objects
    IO=RiverIO('Hydrochron+Gage',obsFname=fname,slope_opt=slope_opt)
    D=Domain(IO.ObsData)    
    
    if area_opt=='Finite Difference':
        dAOpt=0
        CalcAreaFitOpt=0
    elif area_opt=='Fluvial Hypsometry':
        dAOpt=1
        CalcAreaFitOpt=3
    else:
        print('  unknown area option',file=logoutput)
        return None
    
    Obs=ReachObservations(D,IO.ObsData,ConstrainHWSwitch=constrainhw,CalcAreaFitOpt=CalcAreaFitOpt,dAOpt=dAOpt,Verbose=Verbose)        
    Truth=ReachTruth(IO.TruthData)
    
    # set up reach dictionary
    ReachDict={}
    ReachDict['dA']=Obs.dA[0,:]
    ReachDict['w']=Obs.w[0,:]
    ReachDict['S']=Obs.S[0,:]
    ReachDict['H']=Obs.h[0,:]            
    ReachDict['Qtrue']=Truth.Q[0,:]
    
    # calibrate
    FlowLaw=getFlowLaw(flowlawname)    
    
    # return FlowLaw,ReachDict['Qtrue'] #
    
    Cal=FlowLawCalibration(D,ReachDict['Qtrue'],FlowLaw)
    Cal.CalibrateReach(verbose=False,suppress_warnings=False,lossfun=lossfun)
    
    return Cal

def SetOptions(expdf,idx):
    expsid=expdf.iloc[idx]['expsid']
    expid=expdf.iloc[idx]['expid']
    print('running',expid)
    print('  domain:',expdf.iloc[idx]['reachdomain'])
    print('  swot data source:',expdf.iloc[idx]['swotsource'])
    print('  slope data element:',expdf.iloc[idx]['slopedata'])
    print('  minimum slope applied as filter:',expdf.iloc[idx]['slopeminimum'])
    print('  slope consistency check:',expdf.iloc[idx]['slopeconsistencycheck'])
    print('  area option:',expdf.iloc[idx]['areaopt'])
    print('  constrain height-width option:',expdf.iloc[idx]['constrainhw'])
    print('  flow law:',expdf.iloc[idx]['flowlaw'])
    print('  dark frac maximum:',expdf.iloc[idx]['darkfracmax'])
    print('  loss function:',expdf.iloc[idx]['lossfun'])
        
    # set data source
    if expdf.iloc[idx]['swotsource']=='hydrochron':
        SWOTSource='Hydrochron' 
        fname_pvd=None
    elif expdf.iloc[idx]['swotsource']=='ADT-pvd':
        # ADT pseudo version D
        SWOTSource='ADT'
        fname_pvd='/Users/mtd/Data/SWOT/Pseudo-Version-D/swot-adt-data/pseudo-version-d-reach.csv'
        
    # set slope option
    slope_data_element=expdf.iloc[idx]['slopedata']
    slope_data_element       
    
    # set slope threhsold minimum
    slope_min=expdf.iloc[idx]['slopeminimum']
    if np.isnan(slope_min):    
        slope_min=None
    # print(slope_min)        
    
    # set whether or not to apply the slope consistency check
    #    if set to 0, then uses Colin's metho
    #    if < 0, doesn't do anything
    #    if > 0, uses this as the threshold for replacing low slopes
    slope_const=expdf.iloc[idx]['slopeconsistencycheck']
    slope_const   
    
    # area option
    if expdf.iloc[idx]['areaopt']=='fd':
        area_option='Finite Difference'
    elif expdf.iloc[idx]['areaopt']=='fh':
        area_option='Fluvial Hypsometry'
    # print(area_option)  
    
    # constrain height width option
    constrainhw=expdf.iloc[idx]['constrainhw']
    # constrainhw    
    
    flowlaw=expdf.iloc[idx]['flowlaw']
    # flowlaw
    
    if np.isnan(expdf.iloc[idx]['darkfracmax']):
        darkfracmax=None
    else:
        darkfracmax=expdf.iloc[idx]['darkfracmax']
    # print(darkfracmax)    
    
    reachdomain=expdf.iloc[idx]['reachdomain']
    # reachdomain    
 
    # set loss function
    lossfun=expdf.iloc[idx]['lossfun']
    
    return expsid,expid,SWOTSource,fname_pvd,slope_data_element,slope_min,slope_const,area_option,constrainhw,flowlaw,darkfracmax,reachdomain,lossfun
