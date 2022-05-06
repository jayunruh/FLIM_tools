# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:39:19 2020

@author: Jay Unruh, Stowers Institute
License: GPL_v2: http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
"""
from numba import jit
import numpy as np

class linleastsquares():

    def __init__(self,indvars,addbaseline=False,startfit=0,endfit=-1):
        """
        indvars is an nvars x npts numpy array of independent variables to fit to
        endfit is actually endfit+1
        """
        if(not addbaseline):
            self.nindvars=indvars.shape[0]
            self.npts=indvars.shape[1]
            self.indvars=indvars
        else:
            self.nindvars=indvars.shape[0]+1
            self.npts=indvars.shape[1]
            baseindvars=np.ones((1,self.npts))
            self.indvars=np.concatenate((baseindvars,indvars),axis=0)
            #print(self.indvars.shape)
        self.startfit=startfit
        if(endfit<0 or endfit>self.npts): self.endfit=self.npts
        else: self.endfit=endfit
        self.npts=self.endfit-self.startfit

    def fitdata(self,data,weights1=None):
        """
        here is the linear least squares implementation
        """
        jacobian=np.zeros((self.nindvars,self.nindvars),dtype=np.double)
        jvector=np.zeros((self.nindvars),dtype=np.double)
        weights=np.ones(self.npts,dtype=np.double)
        if(weights1 is not None):
            for i in range(self.npts): weights[i]=weights1[i]
        for i in range(self.nindvars):
            for j in range(i+1):
                jacobian[i,j]=np.sum(self.indvars[i,self.startfit:self.endfit]*self.indvars[j,self.startfit:self.endfit]*weights[self.startfit:self.endfit])
                if(i!=j): jacobian[j,i]=jacobian[i,j]
            jvector[i]=np.sum(self.indvars[i,self.startfit:self.endfit]*data[self.startfit:self.endfit]*weights[self.startfit:self.endfit])
        try:
            #if(verbose): print(jacobian)
            outcoef=np.linalg.solve(jacobian,jvector)
            #if(verbose): print(dparams)
        except:
            outcoef=np.zeros((self.nindvars))
            print('singular matrix encountered')
        return outcoef

    def fitmultidata(self,data,weights1=None):
        """
        this version automates analysis of multiple data sets in axis 1 of the data array
        not sure if this works yet
        """
        nmulti=data.shape[0]
        jacobian=np.zeros((self.nindvars,self.nindvars),dtype=np.double)
        weights=np.ones(self.npts,dtype=np.double)
        if(weights1 is not None):
            for i in range(self.npts): weights[i]=weights1[i]
        for i in range(self.nindvars):
            for j in range(i+1):
                jacobian[i,j]=np.sum(self.indvars[i,self.startfit:self.endfit]*self.indvars[j,self.startfit:self.endfit]*weights[self.startfit:self.endfit])
                if(i!=j): jacobian[j,i]=jacobian[i,j]
        try:
            jinv=np.linalg.inv(jacobian)
        except:
            outcoef=np.zeros((nmulti,self.nindvars),dtype=np.double)
            print('singular matrix encountered')
            return outcoef
        jvectors=[(data[:,self.startfit:self.endfit]*self.indvars[j,self.startfit:self.endfit]*weights[self.startfit:self.endfit]).sum(axis=1) for j in range(self.nindvars)]
        jvectors=np.array(jvectors)
        return np.matmul(jinv,jvectors)

    def getfiterrors(self,data,weights1=None):
        """
        this does linear least squares and outputs standard errors in the coefficients
        """
        jacobian=np.zeros((self.nindvars,self.nindvars),dtype=np.double)
        jvector=np.zeros((self.nindvars),dtype=np.double)
        weights=np.ones(self.npts,dtype=np.double)
        if(weights1 is not None):
            for i in range(self.npts): weights[i]=weights1[i]
        for i in range(self.nindvars):
            for j in range(i+1):
                jacobian[i,j]=np.sum(self.indvars[i,self.startfit:self.endfit]*self.indvars[j,self.startfit:self.endfit]*weights[self.startfit:self.endfit])
                if(i!=j): jacobian[j,i]=jacobian[i,j]
            jvector[i]=np.sum(self.indvars[i,self.startfit:self.endfit]*data[self.startfit:self.endfit]*weights[self.startfit:self.endfit])
        try:
            #if(verbose): print(jacobian)
            minv=np.linalg.inv(jacobian)
            outcoef=np.matmul(minv,jvector)
            c2=self.get_c2(outcoef,data,weights)
            se=np.sqrt(np.diag(minv)*c2)
            #if(verbose): print(dparams)
        except:
            outcoef=np.zeros((self.nindvars),dtype=np.double)
            se=np.zeros((self.nindvars),dtype=np.double)
            print('singular matrix encountered')
        return outcoef,se

    def get_c2(self,coef,data,weights):
        """
        this calculates the chi squared from the coefficients and data
        """
        fit=self.get_fit(coef)
        c2=np.sum(weights[self.startfit:self.endfit]*(fit[self.startfit:self.endfit]-data[self.startfit:self.endfit])*(fit[self.startfit:self.endfit]-data[self.startfit:self.endfit]))
        return c2/(self.endfit-self.startfit+1-self.nindvars)

    def get_fit(self,coef):
        """
        here we get the fit from the coefficients
        get the whole fit including values outside the start and end
        """
        fit=np.zeros((self.npts),dtype=np.double)
        for i in range(self.npts):
            fit[i]=np.sum(coef*self.indvars[:,i])
        return fit

@jit(nopython=True)
def getAmpOffset(function,data):
    sumx2=np.sum(function*function)
    sumx=np.sum(function)
    sumy=np.sum(data)
    sumxy=np.sum(function*data)
    dlength=float(len(data))
    if(sumx2>0.0):
        divider=dlength*sumx2-sumx*sumx
        off=(sumx2*sumy-sumx*sumxy)/divider
        amp=(dlength*sumxy-sumx*sumy)/divider
    else:
        amp=0.0
        off=sumy/dlength
    return amp,off

@jit(nopython=True)
def getAmpOffsetC2(func,data,amp,off):
    resid=amp*func+off-data
    c2=np.sum(resid*resid)
    return c2/float(len(data)-2)

@jit(nopython=True)
def getAmpOffsetErrs(function,data):
    sumx2=np.sum(function*function)
    sumx=np.sum(function)
    sumy=np.sum(data)
    sumxy=np.sum(function*data)
    dlength=float(len(data))
    if(sumx2>0.0):
        divider=dlength*sumx2-sumx*sumx
        off=(sumx2*sumy-sumx*sumxy)/divider
        amp=(dlength*sumxy-sumx*sumy)/divider
        c2=np.sum((data-function*amp-off)**2.0)/(dlength-2.0)
        seoff=np.sqrt(c2*sumx2/divider)
        seamp=np.sqrt(c2*dlength/divider)
        r2=np.sum((data-np.mean(data))**2.0)/(dlength-2.0)
        r2=1-c2/r2
    else:
        amp=0.0
        off=sumy/dlength
        seamp=0.0
        seoff=0.0
        c2=np.sum((data-off)**2.0)/(dlength-2.0)
        r2=np.sum((data-np.mean(data))**2.0)/(dlength-2.0)
        r2=1-c2/r2
    return amp,off,seamp,seoff,c2,r2

if __name__ == "__main__":
    #test the linear least squares on some gaussian data
    npts=100
    import random
    import gausfunc
    gf=gausfunc.gausfunc()
    def fitfunc(params):
        func=params[0]+params[1]*gf.get_func(-params[2],npts,1.0,params[3])
        return np.double(func)

    basedata=fitfunc([0.0,1.0,45.0,2.0])
    testdata=fitfunc([1.0,10.0,45.0,2.0])+[random.gauss(0.0,0.5) for i in range(npts)]
    print(testdata)
    fitclass=linleastsquares(np.array([basedata]),addbaseline=True)
    coef,se=fitclass.getfiterrors(testdata)
    print(coef)
    print(se)
