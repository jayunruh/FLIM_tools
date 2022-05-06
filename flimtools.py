# +
"""
copyright Jay Unruh, Stowers Institute, 2022
License: GPL_v2: http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
collection of functions to work with FLIM data
""" 

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import linleastsquares

def readBin(fpath):
    #reads a picoquant bin file
    head=np.fromfile(fpath,dtype=np.uint32,offset=0,count=5)
    width=head[0]
    height=head[1]
    slices=head[3]
    print(head)
    img=np.fromfile(fpath,dtype=np.uint32,offset=20).reshape((height,width,slices))
    img=np.moveaxis(img,2,0)
    print(img.shape)
    return img

def getCircCoords(rad):
    #this gets the relative circle x,y coordinates for a specified radius
    #first get the square grid
    irad=int(rad)+1
    xg,yg=np.meshgrid(np.arange(-irad,irad),np.arange(-irad,irad))
    xg=xg.flatten()
    yg=yg.flatten()
    rad2=rad**2
    dist2=xg**2+yg**2
    coords=np.array([yg[dist2<=rad2],xg[dist2<=rad2]])
    return coords.T

#now measure a point across the stack
def measureCirc(stack,xc,yc,rad,statfunc):
    if(xc<=rad): return None
    if(yc<=rad): return None
    if(xc>=(stack.shape[2]-rad-1)): return None
    if(yc>=(stack.shape[1]-rad-1)): return None
    #get the relative circle coordinates
    ccoords=getCircCoords(rad)
    return statfunc(stack[:,ccoords[:,0]+int(yc),ccoords[:,1]+int(xc)],axis=1)

#output the avg profile phasor values
def calcGS(profile,pshift,mscale,harmonic=1):
    plen=len(profile)
    pshiftscale=pshift*plen/360.0
    return [(mscale*np.cos(2.0*np.pi*harmonic*(np.arange(plen)-pshiftscale)/plen)*profile).sum()/profile.sum(),
            (mscale*np.sin(2.0*np.pi*harmonic*(np.arange(plen)-pshiftscale)/plen)*profile).sum()/profile.sum()]

#high speed phasor calc for entire image
def calcGSimg(stack,pshift,mscale,harmonic=1):
    plen=stack.shape[0]
    pshiftscale=pshift*plen/360.0
    cosprof=mscale*np.cos(2.0*np.pi*harmonic*(np.arange(plen)-pshiftscale)/plen)
    cosprof=cosprof.reshape((plen,1,1))
    sinprof=mscale*np.sin(2.0*np.pi*harmonic*(np.arange(plen)-pshiftscale)/plen)
    sinprof=sinprof.reshape((plen,1,1))
    sumimg=stack.sum(axis=0)
    clipped=sumimg.clip(min=1.0)
    gimg=(stack*cosprof).sum(axis=0)/clipped
    simg=(stack*sinprof).sum(axis=0)/clipped
    return gimg,simg,sumimg

#get the gs histogram presmoothing the g and s images
def getGSHist(gimg,simg,timg,smsigma=1,thresh=1,nbins=64,lims=[0,1,0,1]):
    smg=ndi.gaussian_filter(gimg,sigma=smsigma)
    sms=ndi.gaussian_filter(simg,sigma=smsigma)
    gthresh=smg[timg>thresh]
    sthresh=sms[timg>thresh]
    xthreshcoords,ythreshcoords=np.meshgrid(np.arange(timg.shape[1]),np.arange(timg.shape[0]))
    xthreshcoords=xthreshcoords[timg>thresh].flatten()
    ythreshcoords=ythreshcoords[timg>thresh].flatten()
    gbins=np.linspace(lims[0],lims[1],nbins+1)
    sbins=np.linspace(lims[2],lims[3],nbins+1)
    hist,_,_=np.histogram2d(gthresh,sthresh,[gbins,sbins])
    return hist,gbins,sbins,gthresh,sthresh,xthreshcoords,ythreshcoords

#for histograms it's nice to have a color map with a white background
def getNiceCmap():
    jcmwb=plt.cm.jet(np.arange(256))
    jcmwb[0,:]=1.0
    return colors.ListedColormap(jcmwb)

#make a mask for the selected regions of the histogram
#rect is xc,yc,width,height in histogram units
def getRectHistMask(rect,timg,gthresh,sthresh,xthreshcoords,ythreshcoords):
    #get the rectangle origin
    x=rect[0]-rect[2]/2
    y=rect[1]-rect[3]/2
    #print the box start and end
    print([x,x+rect[2],y,y+rect[3]])
    #create the index filter
    inrect=(gthresh>=x) & (sthresh>=y) & (gthresh<(x+rect[2])) & (sthresh<(y+rect[3]))
    mask=np.zeros(timg.shape,dtype=bool)
    #and mask the image
    mask[ythreshcoords[inrect],xthreshcoords[inrect]]=True
    return mask

#here ummix the entire stack with decay profiles
def unmixStack(stack,profiles):
    #note that the profiles are assumed to be normalized
    lls=linleastsquares.linleastsquares(profiles)
    fractions=lls.fitmultidata(stack.reshape((stack.shape[0],stack.shape[1]*stack.shape[2])).T)
    return fractions.reshape((len(profiles),stack.shape[1],stack.shape[2]))

#here unmix an array of decays (like a carpet, nsamples x decaypoint)
#returns nsamples x coefficients
def unmixDecays(decays,profiles):
    lls=linleastsquares.linleastsquares(profiles)
    return lls.fitmultidata(decays).T

#instead of umixing the whole stack we can instead unmix the histogram pixels
#then map them back to the image
def unmixHist(stack,profiles,gthresh,sthresh,xthreshcoords,ythreshcoords,histthresh=50,nbins=64,lims=[0,1,0,1]):
    #first find out which xy coords correspond to which bins
    scalegcoords=np.floor((gthresh-lims[0])*nbins/(lims[1]-lims[0])).astype(int)
    scalescoords=np.floor((sthresh-lims[2])*nbins/(lims[3]-lims[2])).astype(int)
    decays=np.zeros((nbins,nbins,stack.shape[0]),dtype=float)
    #need to sum the decays in each histogram bin
    for i in range(len(scalegcoords)):
        if((scalegcoords[i]>=0 )& (scalegcoords[i]<nbins) & (scalescoords[i]>=0) & (scalescoords[i]<nbins)):
            decays[scalescoords[i],scalegcoords[i],:]+=stack[:,ythreshcoords[i],xthreshcoords[i]]
    #now unmix them if they have data
    hsums=decays.sum(axis=2)
    lls=linleastsquares.linleastsquares(profiles)
    fractions=lls.fitmultidata(decays.reshape((nbins*nbins,stack.shape[0])))
    fractions=fractions.reshape((len(profiles),nbins,nbins))
    fractions[:,hsums==0]=0.0
    fractions/=fractions.sum(axis=0).clip(min=1)
    proj=stack.sum(axis=0)
    #now map the fractions back to the image for the corresponding pixels
    unmixed=np.zeros((len(profiles),stack.shape[1],stack.shape[2]),dtype=float)
    for i in range(len(scalegcoords)):
        if((scalegcoords[i]>=0 )& (scalegcoords[i]<nbins) & (scalescoords[i]>=0) & (scalescoords[i]<nbins)):
            tfrac=fractions[:,scalescoords[i],scalegcoords[i]]
            #tfrac/=tfrac.sum()
            unmixed[:,ythreshcoords[i],xthreshcoords[i]]=proj[ythreshcoords[i],xthreshcoords[i]]*tfrac
    return unmixed,fractions,decays

#here we instead unmix geometrically
#if there are two points, the position on the axis between the points determines the fraction
#if there are three, the triangle determines the linear combo
#in the case of 4 or more, we simulate a large number of fractions and use them to map our pixel values
def unmixGeom(refpos,sumimg,gthresh,sthresh,xthreshcoords,ythreshcoords,nbins=64,lims=[0,1,0,1]):
    #first find out which gs coords correspond to which bins
    scalegcoords=np.floor((gthresh-lims[0])*nbins/(lims[1]-lims[0])).astype(int)
    scalescoords=np.floor((sthresh-lims[2])*nbins/(lims[3]-lims[2])).astype(int)
    gshist,_,_=np.histogram2d(scalegcoords,scalescoords,[np.arange(nbins+1),np.arange(nbins+1)])
    if(len(refpos)==2):
        fractions=dosfractions(refpos,lims,nbins)
    elif(len(refpos)==3):
        fractions=tripfractions(refpos,lims,nbins)
    else:
        fractions=sim_fractions(refpos,1000000,lims,nbins)
    #finally map the fractions back to the image
    unmixed=np.zeros((len(refpos),sumimg.shape[0],sumimg.shape[1]),dtype=float)
    for i in range(len(scalegcoords)):
        if((scalegcoords[i]>=0 )& (scalegcoords[i]<nbins) & (scalescoords[i]>=0) & (scalescoords[i]<nbins)):
            tfrac=fractions[:,scalescoords[i],scalegcoords[i]]
            #tfrac/=tfrac.sum()
            unmixed[:,ythreshcoords[i],xthreshcoords[i]]=sumimg[ythreshcoords[i],xthreshcoords[i]]*tfrac
    fractions[:,gshist.T==0]=0
    return unmixed,fractions

def dosfractions(refpos,lims=[0,1,0,1],nbins=64):
    #calculate the fraction of every pixel between the x and y points
    g,s=np.meshgrid(np.linspace(lims[0],lims[1],nbins),np.linspace(lims[2],lims[3],nbins))
    fx=(g-refpos[1,0])/(refpos[0,0]-refpos[1,0])
    fy=(s-refpos[1,1])/(refpos[0,1]-refpos[1,1])
    #and the average
    fa=0.5*(fx+fy)
    #handle the edge cases where the points have the same x or y values
    if(np.abs(refpos[0,0]-refpos[1,0])<0.00001): fa=fy
    if(np.abs(refpos[0,1]-refpos[1,1])<0.00001): fa=fx
    fractions=np.array([fa,1.0-fa])
    return fractions

def tripfractions(refpos,lims=[0,1,0,1],nbins=64):
    matrix=np.array([[refpos[0,0],refpos[1,0],refpos[2,0]],
                [refpos[0,1],refpos[1,1],refpos[2,1]],
                [1.0,1.0,1.0]])
    minv=np.linalg.inv(matrix)
    #calculate the fraction of every pixel between the x and y points
    g,s=np.meshgrid(np.linspace(lims[0],lims[1],nbins),np.linspace(lims[2],lims[3],nbins))
    temp=np.array([g.flatten(),s.flatten(),np.ones((nbins*nbins),dtype=float)])
    print(temp.shape)
    temp2=np.matmul(minv,temp).T.reshape((nbins,nbins,3))
    fractions=np.moveaxis(temp2,2,0)
    return fractions
        
#here's our simulation code for random fractions of the components
def sim_fractions(refpos,nsims=1000000,lims=[0,1,0,1],nbins=64):
    #get a bunch of random numbers
    simfrac=np.random.random((nsims,len(refpos)))
    #normalize them into fractions adding up to 1
    simfrac=(simfrac.T/simfrac.sum(axis=1)).T
    #now convert them into gs values based in our reference points
    gvals=(simfrac*refpos[:,0]).sum(axis=1)
    svals=(simfrac*refpos[:,1]).sum(axis=1)
    #finally average all of the fractions from each gs bin
    scalegvals=np.floor((gvals-lims[0])*nbins/(lims[1]-lims[0])).astype(int)
    scalesvals=np.floor((svals-lims[0])*nbins/(lims[1]-lims[0])).astype(int)
    #select the ones that are inbounds
    inbounds=(scalegvals>=0) & (scalesvals>=0) & (scalegvals<nbins) & (scalesvals<nbins)
    #flatten for aggregation purposes and remove points outside of the limits
    scaleflat=scalegvals[inbounds]+scalesvals[inbounds]*nbins
    simfrac=simfrac[inbounds]
    fracflat=np.zeros((nbins*nbins,len(refpos)),dtype=float)
    #now aggregate them
    fracflat[scaleflat]+=simfrac
    #and normalize
    fracflat=(fracflat.T/fracflat.sum(axis=1).clip(min=1)).T
    #and return the unflattened
    return np.moveaxis(fracflat.reshape((nbins,nbins,len(refpos))),2,0)
