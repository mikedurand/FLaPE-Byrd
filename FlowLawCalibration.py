#!/usr/bin/env python3
"""
Created on Thu Jan 21 04:49:20 2021

# -*- coding: utf-8 -*-
@author: mtd
"""
import numpy as np
from scipy import optimize
from numpy import zeros,empty,nan,mean,log,exp,polyfit,array
from ErrorStats import ErrorStats

import matplotlib.pyplot as plt
import warnings
import sys


class FlowLawCalibration:
    def __init__(self,D,Qtrue,FlowLaw):
        self.D=D
        self.Qtrue=Qtrue
        self.FlowLaw=FlowLaw
        
        self.param_est=[]
        self.success=[]
        self.Qhat=[]
        self.n=[]
        self.n_t=[]
        self.A_t=[]
        self.n_numerator=[]
        self.Performance={}

    def CalibrateReach(self, verbose=True, optmethod='L-BFGS-B', suppress_warnings=False, lossfun='linear'):
        if suppress_warnings:
            warnings.filterwarnings("ignore")
    
        self.success = np.zeros(1, dtype=bool)
        self.Qhat = np.zeros((1, self.D.nt))
    
        init_params = self.FlowLaw.GetInitParams()
        fl_param_bounds = self.FlowLaw.GetParamBounds()



    
        # 1. Least squares (AHGW only)
        self.success = False
        ls_success = False
        if self.FlowLaw.name == 'AHGW':
            a, b = self.LeastSquaresFit(self.FlowLaw.W, self.Qtrue)
            if (fl_param_bounds[0][0] <= a <= fl_param_bounds[0][1]) and (fl_param_bounds[1][0] <= b <= fl_param_bounds[1][1]):
                ls_success = True
                self.success = True
                print('... AHGW: polyfit worked!')
    
        # 2. Optimization setup
        nparams = len(init_params)
        lb = np.zeros(nparams)
        ub = np.zeros(nparams)
        for i in range(nparams):
            lb[i] = fl_param_bounds[i][0]
            ub[i] = fl_param_bounds[i][1]
    
        param_bounds = optimize.Bounds(lb, ub)

        
    
        # 3. least_squares
        if not self.success:
 
            # print("\n===== DEBUGGING least_squares input =====")
            # print("Initial parameters (init_params):", init_params)
        
            # try:
            #     Qhat_debug = self.FlowLaw.CalcQ(init_params)
            #     residual_debug = Qhat_debug - self.Qtrue
            #     print("Qhat (from CalcQ):", Qhat_debug)
            #     print("Qtrue:", self.Qtrue)
            #     print("Residuals (Qhat - Qtrue):", residual_debug)
            #     print("Are all residuals finite?", np.all(np.isfinite(residual_debug)))
            #     sys.stdout.flush() 
            # except Exception as e:
            #     print("Error when calculating Qhat or residuals:", e)
            # print("===== DEBUGGING: Check Qhat Inputs =====")
            # A0 = init_params[1]
            # A_t = A0 + self.FlowLaw.dA
            # W = self.FlowLaw.W
            # S = self.FlowLaw.S
            # try:
            #     print("A_t:", A_t)
            #     print("W:", W)
            #     print("S:", S)
            #     print("Any A_t <= 0:", np.any(A_t <= 0))
            #     print("Any W <= 0:", np.any(W <= 0))
            #     print("Any S < 0:", np.any(S < 0))
            #     #sys.stdout.flush()
            # except Exception as e:
            #     print("Exception when printing arrays:", e)


            
            res = optimize.least_squares(self.ObjectiveFuncRes, init_params, args=([self.Qtrue]), bounds=param_bounds, loss=lossfun)
            if res.success:
                if verbose:
                    print('... the least_squares solution worked')
                self.success = True
    
        # 4. fallback to minimize
        if not self.success:
            print('... least_squares failed. Now trying with L-BFGS-B')
            res = optimize.minimize(fun=self.ObjectiveFunc, x0=init_params, args=(self.Qtrue),
                                    bounds=param_bounds, method=optmethod, jac='none')
            if res.success:
                print('... L-BFGS-B succeeded!')
                self.success = True
    
        if not self.success:
            print('... trying trust-constr')
            res = optimize.minimize(fun=self.ObjectiveFunc, x0=init_params, args=(self.Qtrue),
                                    bounds=param_bounds, method='trust-constr', jac='none')
            if res.success:
                print('... trust-constr succeeded!')
                self.success = True
    
        if not self.success:
            param_bounds = optimize.Bounds(lb, ub, keep_feasible=True)
            res = optimize.minimize(fun=self.ObjectiveFunc, x0=init_params, args=(self.Qtrue),
                                    bounds=param_bounds, method=optmethod,
                                    jac='none', options={'disp': verbose, 'maxiter': 1e4, 'verbose': 0})
            if res.success:
                print('... backup algorithm succeeded!')
                self.success = True
    
        if not self.success:
            res = optimize.minimize(fun=self.ObjectiveFunc, x0=init_params, args=(self.Qtrue),
                                    bounds=param_bounds, method=optmethod, jac='none',
                                    options={'disp': verbose, 'maxiter': 1e4, 'verbose': 0, 'finite_diff_rel_step': 1.0})
            if res.success:
                print('... backup with larger step size succeeded!')
                self.success = True
    
        # 5. Handle results
        if not self.success:
            print('FlowLawCalibration: Optimize Failed! Setting flow law parameters to NaN')
            self.param_est = np.empty(len(fl_param_bounds))
            self.param_est[:] = np.nan
        else:
            self.param_est = np.array([a, b]) if ls_success else res.x
    
        # 6. Compute Qhat and n
        self.Qhat = self.FlowLaw.CalcQ(self.param_est)
        self.n = self.FlowLaw.CalcN(self.param_est)
    
        # 7. Compute n_t using Manning’s equation (your definition)
        try:
            A0 = self.param_est[1]
            A_t = A0 + self.FlowLaw.dA
            A_t[A_t <= 0] = np.nan      
            W_t = self.FlowLaw.W
            W_t[W_t <= 0] = np.nan     
            S = self.FlowLaw.S
            S[S <= 0] = np.nan         
            Q = np.where(self.Qtrue == 0, np.nan, self.Qtrue)
            numerator = A_t ** (5 / 3) * W_t ** (-2 / 3) * np.sqrt(S)
            
            self.A_t = A_t
            self.n_numerator = numerator
            self.n_t = numerator / Q
        except Exception as e:
            print("Failed to compute n_t:", e)
            self.n_t = np.full_like(self.Qtrue, np.nan)
    
        # 8. Compute stats
        self.Performance = ErrorStats(self.Qtrue, self.Qhat, self.D)
        self.Performance.CalcErrorStats()


    
    def ObjectiveFunc(self,params,Q):      
        Qhat=self.FlowLaw.CalcQ(params)
        y=sum((Qhat-Q)**2)
        return y

    def ObjectiveFuncRes(self,params,Q):      
        Qhat=self.FlowLaw.CalcQ(params)
        y=Qhat-Q
        return y

    def LeastSquaresFit(self,O,Q):
        # O = observations, either H or W
        # Q = discharge

        y=log(Q)
        x=log(O)
    
        p=polyfit(x,y,1)
    
        a=exp(p[1])
        b=p[0]
    
        return a,b


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
