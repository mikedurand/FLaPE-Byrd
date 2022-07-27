#!/usr/bin/env python3
"""
Created on Thu Jan 21 04:49:20 2021

# -*- coding: utf-8 -*-
@author: mtd
"""

from scipy import optimize
from numpy import zeros,empty,nan,mean
from ErrorStats import ErrorStats

import matplotlib.pyplot as plt
import warnings

class FlowLawCalibration:
    def __init__(self,D,Qtrue,FlowLaw):
        self.D=D
        self.Qtrue=Qtrue
        self.FlowLaw=FlowLaw
        
        self.param_est=[]
        self.success=[]
        self.Qhat=[]
        self.Performance={}

    def CalibrateReach(self,verbose=True,optmethod='trust-constr',suppress_warnings=False):     
  
        if suppress_warnings:
            warnings.filterwarnings("ignore")
        
        # self.param_est=zeros( (self.D.nR,2) )
        self.success= zeros( 1, dtype=bool )
        self.Qhat=zeros( (1,self.D.nt) )    
                   
        init_params=self.FlowLaw.GetInitParams()
        fl_param_bounds=self.FlowLaw.GetParamBounds()         

        np=len(init_params)
  
        lb=zeros(np,)
        ub=zeros(np,)
        if np > 1:
            for i in range(np):
                lb[i]=fl_param_bounds[i][0] 
                ub[i]=fl_param_bounds[i][1] 
        else: 
            lb[0]=fl_param_bounds[0]
            ub[0]=fl_param_bounds[1]

        param_bounds=optimize.Bounds(lb,ub)

        #print('Qtrue=',self.Qtrue)
        #print('width=',self.FlowLaw.W)
        #print('initial flow law parameters=',init_params)
        #print('Jacobian at initial parameters=',self.FlowLaw.Jacobian(init_params,self.Qtrue) )

        #note can set verbose option to 3 for debugging
        res = optimize.minimize(fun=self.ObjectiveFunc,
                                x0=init_params,
                                args=(self.Qtrue),
                                bounds=param_bounds,
                                method=optmethod,
                                jac=self.FlowLaw.Jacobian,
                                options={'disp':verbose,'maxiter':1e4,'verbose':0})

        if not res.success:
            #retry with bounds required to stay feasible
            param_bounds=optimize.Bounds(lb,ub,keep_feasible=True)
            
            res = optimize.minimize(fun=self.ObjectiveFunc,
                                x0=init_params,
                                args=(self.Qtrue),
                                bounds=param_bounds,
                                method=optmethod,
                                jac=self.FlowLaw.Jacobian,
                                options={'disp':verbose,'maxiter':1e4,'verbose':0})
        
        if not res.success:
             #retry with a larger step size
             res = optimize.minimize(fun=self.ObjectiveFunc,
                                x0=init_params,
                                args=(self.Qtrue),
                                bounds=param_bounds,
                                method=optmethod,
                                jac=self.FlowLaw.Jacobian,
                                options={'disp':verbose,'maxiter':1e4,'verbose':0,'finite_diff_rel_step':1.0})
 
        self.success=res.success

        if not self.success:
            print('FlowLawCalibration: Optimize Failed! Setting flow law parameters to nan')
            self.param_est=empty( len(fl_param_bounds), )
            self.param_est[:]=nan
        else:
            self.param_est=res.x
            
        self.Qhat=self.FlowLaw.CalcQ(self.param_est)      
        
        self.Performance=ErrorStats(self.Qtrue,self.Qhat,self.D)
        self.Performance.CalcErrorStats()

    
    def ObjectiveFunc(self,params,Q):      
        Qhat=self.FlowLaw.CalcQ(params)
        y=sum((Qhat-Q)**2)
        return y


    def PlotTimeseries(self,PlotTitle='',SaveFilename='',ShowLegend=True):
        fig,ax = plt.subplots()
        ax.plot(self.D.t.T,self.Qtrue,label='gage',marker='o')
        ax.plot(self.D.t.T,self.Qhat,label='estimate',marker='+')        
        plt.title(PlotTitle)
        plt.xlabel('Observation Number')
        plt.ylabel('Discharge $m^3/s$')
        if ShowLegend:
            plt.legend()        
        if SaveFilename:
            plt.savefig(SaveFilename)
        plt.show()
    def PlotScatterplot(self):
        fig,ax = plt.subplots()
        ax.scatter(self.Qtrue,self.Qhat,marker='o')        
        y_lim = ax.get_ylim()
        x_lim = ax.get_xlim()
        onetoone=[0,0]
        onetoone[0]=min(y_lim[0],x_lim[0])
        onetoone[1]=max(y_lim[1],x_lim[1])
        ax.plot([onetoone[0],onetoone[1]],[onetoone[0],onetoone[1]])
        plt.title('Discharge scatterplot')
        plt.xlabel('True Discharge $m^3/s$')
        plt.ylabel('Estimated Discharge $m^3/s$')      
        plt.show()   
    def PlotScatterQW(self,logscale=False):
        fig,ax = plt.subplots()
        ax.scatter(self.FlowLaw.W,self.Qtrue,marker='o')        
        plt.xlabel('Measured width m')
        plt.ylabel('True Discharge $m^3/s$')
        if logscale:
             ax.set_yscale('log')
             ax.set_xscale('log')
        plt.show()   
    def PlotScatterQH(self):
        fig,ax = plt.subplots()
        ax.scatter(self.FlowLaw.H,self.Qtrue,marker='o')        
        plt.xlabel('Measured WSE m')
        plt.ylabel('True Discharge m^3/s')
        plt.show()   
