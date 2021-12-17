#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:49:40 2021
@author: mtd
"""

from numpy import reshape,concatenate,zeros,ones,triu,empty,arctan,tan,pi,std,mean,sqrt,var,cov
from scipy import stats
import matplotlib.pyplot as plt
import copy

class ReachObservations:    
        
    def __init__(self,D,RiverData,ConstrainHWSwitch=False):
        
        self.D=D    
        self.ConstrainHWSwitch=ConstrainHWSwitch
        
        
        # assign data from input dictionary
        self.h=RiverData["h"]           
        self.w=RiverData["w"]
        self.S=RiverData["S"]
        self.h0=RiverData["h0"]
        self.sigh=RiverData["sigh"]
        self.sigw=RiverData["sigw"]
        self.sigS=RiverData["sigS"]    
        
        # constrain heights and widths to be self-consistent
        #      note - h,w data only overwritten if switch set to true
        self.ConstrainHW()

        #%% create resahepd versions of observations
        self.hv=reshape(self.h, (self.D.nR*self.D.nt,1) )
        self.Sv=reshape(self.S, (self.D.nR*self.D.nt,1) )
        self.wv=reshape(self.w, (self.D.nR*self.D.nt,1) )
        
        DeltaAHat=empty( (self.D.nR,self.D.nt-1) )
        self.DeltaAHatv = self.calcDeltaAHatv(DeltaAHat)
        self.dA= concatenate(  (zeros( (self.D.nR,1) ), DeltaAHat @ triu(ones( (self.D.nt-1,self.D.nt-1) ),0)),1 )
        self.dAv=self.D.CalcU() @ self.DeltaAHatv

    def calcDeltaAHatv(self, DeltaAHat):
        
        for r in range(0,self.D.nR):
            for t in range(0,self.D.nt-1):
                DeltaAHat[r,t]=(self.w[r,t]+self.w[r,t+1])/2 * (self.h[r,t+1]-self.h[r,t])
         
        # changed how this part works compared with Matlab, avoiding translating calcU
        return reshape(DeltaAHat,(self.D.nR*(self.D.nt-1),1) )
    
    def ConstrainHW(self):
        
        self.hobs=copy.deepcopy(self.h[0,:])
        self.wobs=copy.deepcopy(self.w[0,:])

        #range-normalize data
        x=self.hobs
        y=self.wobs
        x_range=max(x)-min(x)        
        x_mean=mean(x) 
        xn=(x-x_mean)/x_range
        y_range=max(y)-min(y)        
        y_mean=mean(y) 
        yn=(y-y_mean)/y_range
        
        #[m,b]=self.FitLOC(xn,yn)
        [m,b]=self.FitEIV(xn,yn)
        mo=-tan(pi/2-arctan(m))
 
        #projet w,h onto LOC
        hhatn=(yn-mo*xn-b)/(m-mo)
        whatn=m*hhatn+b

        #un-normalize data
        hhat=hhatn*x_range+x_mean
        what=whatn*y_range+y_mean

        hres=self.h[0,:]-hhat
        wres=self.w[0,:]-what

        self.stdh_LOChat=std(hres)
        self.stdw_LOChat=std(wres)

        if self.ConstrainHWSwitch:
             self.h[0,:]=hhat
             self.w[0,:]=what

    def plotHW(self):
        fig,ax = plt.subplots()
        
        if self.ConstrainHWSwitch:
            ax.scatter(self.hobs,self.wobs,marker='o')   
            ax.scatter(self.h[0,:],self.w[0,:],marker='o')   
        else: 
            ax.scatter(self.h[0,:],self.w[0,:],marker='o')   
            
        plt.title('WSE vs width for first reach')
        plt.xlabel('WSE, m')
        plt.ylabel('Width, m')      
        plt.show() 
        
    def plotdA(self):
        fig,ax = plt.subplots()
        ax.plot(self.D.t.T,self.dA[0,:])        
            
        plt.title('dA timeseries')
        plt.xlabel('Time, days')
        plt.ylabel('dA, m^2')      
        plt.show()       
        
    def plotHdA(self):
        fig,ax = plt.subplots()
        
        ax.scatter(self.h[0,:],self.dA[0,:],marker='o')   
            
        plt.title('dA vs WSE for first reach')
        plt.xlabel('WSE, m')
        plt.ylabel('dA, m')      
        plt.show()         

    def FitLOC(self,x,y):
        #references from Statistical Methods in Water Resources, by Helsel &
        #Hirsch, 1992. 

        sx=std(x)
        sy=std(y)

        mx=mean(x)
        my=mean(y)

        n=len(x)

        r=1/(n-1) * sum( (x-mx)/sx * (y-my)/sy ) # 8.6: Pearson's r

        b1=r*sy/sx #text below 10.7
        b1prime=1/r*sy/sx #text below 10.9

        b1doubleprime=sqrt(b1*b1prime) #slope: text above 10.10

        b0doubleprime=my-b1doubleprime*mx #intercept: comparing 10.10 with 10.9

        return b1doubleprime, b0doubleprime

    def FitEIV(self,x,y):
        #this is derived from fuller, for the ratio of variances model 1.3.7

        #compute sample variances and covariance
        mXX=var(x)
        mYY=var(y)
        Sigma=cov(x,y)
        mXY=Sigma[0,1]
 
        #this option has delta set to 1
        #beta1hat=((mYY-mXX)+( (mYY-mXX)**2 + 4*mXY**2   )**0.5 ) / (2*mXY)
        
        #this option lets you specify delta
        delta=1.0
        beta1hat=((mYY-delta*mXX)+( (mYY-delta*mXX)**2 + 4*delta*mXY**2   )**0.5 ) / (2*mXY)

        beta0hat=mean(y)-beta1hat*mean(x)

        return beta1hat, beta0hat
