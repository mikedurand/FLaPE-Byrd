#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:09:29 2021

@author: mtd
"""

from numpy import array,diff,ones,reshape,empty    

class RiverIO:
    # def __init__(self,IOtype,obsFname):
    def __init__(self,IOtype,**fnames):        
        self.type=IOtype        
        self.ObsData={}    
        self.TruthData={}
        
        if self.type == 'MetroManTxt':
            if 'obsFname' in fnames.keys():
                self.obsFname=fnames["obsFname"]
                self.ReadMetroManObs()
            if 'truthFname' in fnames.keys():
                self.truthFname=fnames["truthFname"]
                self.ReadMetroManTruth()                
        else:
            print("RiverIO: Undefined observation data format specified. Data not read.")
        
    
    
    def ReadMetroManObs(self):
        # Read observation file in MetroMan text format        
        fid=open(self.obsFname,"r")
        infile=fid.readlines()
        fid.close()   
        
        # read domain
        self.ObsData["nR"]=eval(infile[1])
                
        buf=infile[3]; buf=buf.split(); self.ObsData["xkm"]=array(buf,float)
        buf=infile[5]; buf=buf.split(); self.ObsData["L"]=array(buf,float)
        self.ObsData["nt"]=eval(infile[7]);
        buf=infile[9]; buf=buf.split(); self.ObsData["t"]=array([buf],float)
        
        #note: move this line to ReachObservations...
        self.ObsData["dt"]=reshape(diff(self.ObsData["t"]).T*86400 * ones((1,self.ObsData["nR"])),(self.ObsData["nR"]*(self.ObsData["nt"]-1),1))
        
        # #specify variable sizes
        self.ObsData["h"]=empty(  (self.ObsData["nR"],self.ObsData["nt"]) ) #water surface elevation (wse), [m]
        self.ObsData["h0"]=empty( (self.ObsData["nR"],1)  ) #initial wse, [m]
        self.ObsData["S"]=empty(  (self.ObsData["nR"],self.ObsData["nt"]) ) #water surface slope, [-]
        self.ObsData["w"]=empty(  (self.ObsData["nR"],self.ObsData["nt"]) ) #river top width, [m]     
        self.ObsData["sigh"]=[] #wse uncertainty standard deviation [m]
        self.ObsData["sigS"]=[] #slope uncertainty standard deviation [-]
        self.ObsData["sigW"]=[] #width uncertainty standard deviation [m]
        
        #%% read observations   
        for i in range(0,self.ObsData["nR"]):
            buf=infile[i+11]; buf=buf.split(); self.ObsData["h"][i,:]=array(buf,float)
        
        buf=infile[12+self.ObsData["nR"]]; buf=buf.split(); self.ObsData["h0"]=array(buf,float)
        
        for i in range(0,self.ObsData["nR"]):
            buf=infile[14+self.ObsData["nR"]+i]; buf=buf.split(); self.ObsData["S"][i,:]=array(buf,float)/1e5; #convert cm/km -> m/m
        for i in range(0,self.ObsData["nR"]):
            buf=infile[15+self.ObsData["nR"]*2+i]; buf=buf.split(); self.ObsData["w"][i,:]=array(buf,float)
        self.ObsData["sigS"]=eval(infile[16+self.ObsData["nR"]*3])/1e5; #convert cm/km -> m/m
        self.ObsData["sigh"]=eval(infile[18+self.ObsData["nR"]*3])/1e2; #convert cm -> m
        self.ObsData["sigw"]=eval(infile[20+self.ObsData["nR"]*3] )
        
    def ReadMetroManTruth(self):
        
        if not self.ObsData:
            print("RiverIO/ReadMetroManTruth: Canot read truth file if obs data not read in. Truth data not read.")
            return
        
        fid=open(self.truthFname,"r")
        infile=fid.readlines()
        fid.close()  
        
           
        buf=infile[1]; buf=buf.split(); self.TruthData["A0"]=array(buf,float)
        buf=infile[3]; self.TruthData["q"]=buf #not fully implemented; only affects MetroMan plotting routines
        buf=infile[5]; self.TruthData["n"]=buf #not fully implemented; only affects MetroMan plotting routines

        self.TruthData["Q"]=empty( (self.ObsData["nR"],self.ObsData["nt"])  ) #discharge [m^3/s]
        # reading dA, h, and W truth variables not yet implemented
        # self.dA=empty( (D.nR,D.nt)  ) #cross-sectional area change [m^3/s]
        # self.W=empty( (D.nR,D.nt)  ) #width [m]
        # self.h=empty( (D.nR,D.nt)  ) #wse [m]
        
        for i in range(0,self.ObsData["nR"]):
            buf=infile[i+7]; buf=buf.split(); self.TruthData["Q"][i,:]=array(buf,float) 
            # reading dA, h, and W truth variables not yet implemented
            # buf_dA=infile[i+8+D.nR]; buf_dA=buf_dA.split(); Tru.dA[i,:]=array(buf_dA,float)
            # buf_h=infile[i+9+2*D.nR]; buf_h=buf_h.split(); Tru.h[i,:]=array(buf_h,float)
            # buf_W=infile[i+10+3*D.nR]; buf_W=buf_W.split(); Tru.W[i,:]=array(buf_W,float)
     
    def SubSelectData(self,iUse):
       self.ObsData["nt"]=len(iUse)           
       self.ObsData["h"]= self.ObsData["h"][:,iUse]
       self.ObsData["w"]= self.ObsData["w"][:,iUse]
       self.ObsData["S"]= self.ObsData["S"][:,iUse]
       self.ObsData["t"]= self.ObsData["t"][:,iUse]
       self.TruthData["Q"]= self.TruthData["Q"][:,iUse]
        
