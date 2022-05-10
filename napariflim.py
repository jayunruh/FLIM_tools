"""
copyright Jay Unruh, Stowers Institute, 2022
License: GPL_v2: http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
interactive napari widget for phasor histogram exploration
"""

import napari
import flimtools
from magicgui import magicgui
from napari.types import LabelsData
from napari_matplotlib.base import NapariMPLWidget
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np

class PhasorWidget(NapariMPLWidget):
    n_layers_input=0
    
    def __init__(self, napari_viewer: napari.viewer.Viewer,gshist,gbins,sbins,pts=None,ptsizes=None,lims=[0,1,0,1]):
        super().__init__(napari_viewer)
        self.axes = self.canvas.figure.subplots()
        self.update_layers(None)
        self.gbins=gbins
        self.sbins=sbins
        self.gshist=gshist
        self.pts=np.array(pts)
        self.ptsizes=ptsizes
        self.lims=lims
        self.mult=1.0

    def draw(self) -> None:
        """
        Clear the axes and scatter the currently selected layers.
        """
        self.axes.clear()
        self.axes.pcolormesh(self.gbins,self.sbins,self.gshist.T,cmap=flimtools.getNiceCmap(),
                             vmin=0,vmax=self.gshist.max()/self.mult)
        self.axes.add_patch(patches.Arc((0.5,0.0),1.0,1.0,theta1=0.0,theta2=180.0))
        if self.pts is not None:
            #size=0.5*sum(self.ptsizes*60)
            #self.axes.plot(self.pts[:,0],self.pts[:,1],'rs',mfc='none',ms=size)
            ts=self.ptsizes[0]
            tc=self.pts[0]-0.5*ts
            self.axes.add_patch(patches.Rectangle(tc,ts,ts,ec='red',fc='None'))
            ts=self.ptsizes[1]
            tc=self.pts[1]-0.5*ts
            self.axes.add_patch(patches.Rectangle(tc,ts,ts,ec='red',fc='None'))
        self.axes.set_xlim(self.lims[0],self.lims[1])
        self.axes.set_ylim(self.lims[2],self.lims[3])
        self.axes.set_xlabel('G')
        self.axes.set_ylabel('S')
        self.canvas.draw()
        
    def clear(self) -> None:
        self.axes.clear()

def startNapariWidget(gimg,simg,timg,viewer=None,smsigma=2,thresh=3,lims=[0,1,0,1],npts=2,points=None,ptsizes=None):
    '''
    makes a napari widget that updates masks based on the smoothed histogram
    '''
    tpts=points
    if points is None:
        if npts==1:
            tpts=[[0.5,0.25],[0,0]]
        else:
            tpts=[[0.25,0.25],[0.75,0.25]]

    tptsizes=ptsizes
    if ptsizes is None:
        tptsizes=np.full(2,0.1)

    if(viewer is None):
        viewer=napari.view_image(timg,name='Intensity')
        
    gshist,gbins,sbins,gthresh,sthresh,xthreshcoords,ythreshcoords=\
            flimtools.getGSHist(gimg,simg,timg,smsigma=smsigma,thresh=thresh,lims=lims)
    pw=PhasorWidget(viewer,gshist,gbins,sbins,tpts,tptsizes)
    viewer.window.add_dock_widget(pw)
    pw.draw()

    #if you set up autocall the controls will update as you change them
    #for a big data set that might take too long
    #@magicgui(auto_call=True,call_button='update_plot')
    @magicgui(call_button='update_labels',
             pt1x={'step':0.05},pt1y={'step':0.05},pt1s={'step':0.05},
             pt2x={'step':0.05},pt2y={'step':0.05},pt2s={'step':0.05},
             smooth={'min':0.0,'step':1.0},thresh={'min':0.0,'step':1.0},
             mult={'min':1.0,'step':1.0}
             )
    def makeLabels(pt1x:float=tpts[0][0],pt1y:float=tpts[0][1],pt1s:float=tptsizes[0],
        pt2x:float=tpts[1][0],pt2y:float=tpts[1][1],pt2s:float=tptsizes[1],
        smooth:float=smsigma,thresh:float=thresh,mult:float=1.0)->LabelsData:
        gshist,gbins,sbins,gthresh,sthresh,xthreshcoords,ythreshcoords=\
            flimtools.getGSHist(gimg,simg,timg,smsigma=smooth,thresh=thresh,lims=lims)

        pw.gshist=gshist
        pw.pts=np.array([[pt1x,pt1y],[pt2x,pt2y]])
        pw.ptsizes=[pt1s,pt2s]
        pw.mult=mult
        pw.clear()
        pw.draw()
        mask1=flimtools.getRectHistMask([pt1x,pt1y,pt1s,pt1s],timg,gthresh,sthresh,xthreshcoords,ythreshcoords)
        mask2=flimtools.getRectHistMask([pt2x,pt2y,pt2s,pt2s],timg,gthresh,sthresh,xthreshcoords,ythreshcoords)
        labels=mask1.astype(int)
        labels[mask2]=2
        return labels
   
    viewer.window.add_dock_widget(makeLabels)
    makeLabels.result_name="Phasor_Masks"
    makeLabels()
    napari.run()
    return viewer
