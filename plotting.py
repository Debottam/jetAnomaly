import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

def jetPlot(jets,selection=1,label=None,axis=None,Eplot=False):
    '''Plot the 2D jet image of the jet in eta-phi space.
    By default, constituent pT will be used for the z-axis.
    Inputs:
        jets: collection of jets in the form [njet,nconstituents,(eta,phi,...,pT,eta)]
    Optional:
        selection: any additional jet based selection
        label: jet label for the axis title
        axis: can pass a matplotlib axis instead of creating a new axis
        Eplot: Use constituent energy on the z-axis instead of pT
    Returns:
        matplotlib axis
    '''

    eta = jets[:,:,0]
    phi = jets[:,:,1]
    njet = eta.shape[0]
    nclusters = eta.shape[1]
    z = jets[:,:,-1] if Eplot else jets[:,:,-2]
    z = z.ravel()*np.repeat(selection,nclusters)
    if axis is None:
        fig,axis = plt.subplots(1,1)
    h = axis.hist2d(eta.ravel(),phi.ravel(),weights=z / njet,bins=(np.arange(-1.525,1.525,0.05),np.arange(-1.525,1.525,0.05)), norm=LogNorm(),cmap='viridis',vmin=1e-8,vmax=1.0)
    plt.colorbar(h[3], ax=axis)
    axis.set_xlabel("$\Delta\eta$",horizontalalignment='right',x=1.0)
    axis.set_ylabel("$\Delta\phi$",verticalalignment='top',y=1.0)
    axis.set_title(label)
    return axis

def projectiontionLS_2D(dim1, dim2, latent_space, *args, **kwargs):
    '''Plot a two dimension latent space projection with marginals showing each dimension.
    Can overlay multiple different datasets by passing more than one latent_space argument.

    Inputs:
        dim1: First LS dimension to plot on x axis
        dim2: Second LS dimension to plot on y axis
        latent_space (latent_space2, latent_space3...): the data to plot
    Optional:
        xrange: specify xrange in form [xmin,xmax]
        yrange: specify xrange in form [ymin,ymax]
        labels: labels in form ['ls1','ls2','ls3'] to put in legend
        Additional options will be passed to the JointGrid __init__ function
    Returns:
        seaborn JointGrid object
    '''    
    if 'xrange' in kwargs:
        xrange=kwargs.get('xrange')
    else:
        xrange=(np.floor(np.quantile(latent_space[:,dim1],0.02)),np.ceil(np.quantile(latent_space[:,dim1],0.98)))
    if 'yrange' in kwargs:
        yrange=kwargs.get('yrange')
    else:
        yrange=(np.floor(np.quantile(latent_space[:,dim2],0.02)),np.ceil(np.quantile(latent_space[:,dim2],0.98)))

    labels = [None]*(1+len(args))
    if 'labels' in kwargs:
        labels = kwargs.get('labels')

    kwargs.pop('xrange',None)
    kwargs.pop('yrange',None)
    kwargs.pop('labels',None)

    g = sns.JointGrid(latent_space[:,dim1],latent_space[:,dim2],xlim=xrange,ylim=yrange,**kwargs)

    # for label in [0,1]:
    sns.kdeplot(latent_space[:,dim1],ax=g.ax_marg_x,legend=False,shade=True,alpha=0.3,label=labels[0])
    sns.kdeplot(latent_space[:,dim2],ax=g.ax_marg_y,vertical=True,legend=False,shade=True,alpha=0.3,label=labels[0])
    sns.kdeplot(latent_space[:,dim1], latent_space[:,dim2],ax=g.ax_joint, shade=True, shade_lowest=False,bw=0.2,alpha=1,label=labels[0])

    i = 1
    for ls in args:
        sns.kdeplot(ls[:,dim1],ax=g.ax_marg_x,legend=False,shade=True,alpha=0.3,label=labels[i])
        sns.kdeplot(ls[:,dim2],ax=g.ax_marg_y,vertical=True,legend=False,shade=True,alpha=0.3,label=labels[i])
        sns.kdeplot(ls[:,dim1], ls[:,dim2],ax=g.ax_joint, shade=True, shade_lowest=False,bw=0.2,alpha=0.4,label=labels[i])
        i += 1

    g.ax_joint.spines['right'].set_visible(True)
    g.ax_joint.spines['top'].set_visible(True)
    g.set_axis_labels('LS Dim. {}'.format(dim1),'LS Dim. {}'.format(dim2))
    if labels[0] is not None:
        g.ax_joint.legend()
    return g

