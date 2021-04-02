import numpy as np
import pandas as pd

def ExyzToEtaPhiPtE(fourvectors):
    '''Convert collection of jet constituent fourvectors from Epxpypz representation to EtaPhiPtE.
    Input array fourvectors must be of shape (nJets,nConstituent,4), with fourvector ordered E,px,py,pz.
    Returns same shape but with fourvector ordered eta,phi,pT,E.'''
    
    nconst = fourvectors.shape[1]
    pt = np.sqrt(np.sum(np.power(fourvectors[:,:,1:3],2),axis=-1))
    p =  np.sqrt(np.sum(np.power(fourvectors[:,:,1:],2),axis=-1))
    eta = -0.5*np.log( (1 - fourvectors[:,:,3] / p)/(1+fourvectors[:,:,3] / p)) #need to implement case of pz == 0 (eta = 0)
    eta[(fourvectors[:,:,-1].mask) | (fourvectors[:,:,-1] == 0)] = 0
    phi = np.arctan2(fourvectors[:,:,2], fourvectors[:,:,1])
    e = fourvectors[:,:,0]
    
    phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
    phi = np.where(phi <= -np.pi, phi + 2*np.pi, phi)

    return np.ma.concatenate([
        eta.reshape(-1,nconst,1),
        phi.reshape(-1,nconst,1),
        pt.reshape(-1,nconst,1),
        e.reshape(-1,nconst,1)
    ],axis=-1)

def transform_jets(jets,jet_pt,jet_e):
    #centre jet on leading pT cluster
    jet_eta = jets[:,0,0]
    jet_phi = jets[:,0,1]

    mask = jets.mask.copy()
    nconst = jets.shape[1]

    #jets = np.ma.masked_where(df.mask,jets)
    jets[:,:,:2] = np.ma.masked_array(np.concatenate([(jets[:,:,0] - jet_eta.reshape(-1,1)).reshape(-1,nconst,1),
                                (jets[:,:,1] - jet_phi.reshape(-1,1)).reshape(-1,nconst,1)], axis=-1),mask=mask[:,:,:2])

    #constraint phi between -pi and pi
    jets[:,:,1] = np.where(jets[:,:,1] < np.pi, jets[:,:,1] + 2*np.pi,jets[:,:,1])
    jets[:,:,1] = np.where(jets[:,:,1] >= np.pi, jets[:,:,1] - 2*np.pi,jets[:,:,1])
    
    #Add fraction pT/E
    fracs = np.concatenate([(jets[:,:,2] / jet_pt.reshape(-1,1)).reshape(-1,nconst,1), (jets[:,:,3] / jet_e.reshape(-1,1)).reshape(-1,nconst,1)],axis=-1)
    fracs = np.ma.masked_array(fracs,mask=mask[:,:,:2])
    jets = np.ma.concatenate([jets,fracs],axis=-1)

    #Rotate subleading cluster
    alpha = - np.arctan2(jets[:,1,1],jets[:,1,0])
    alpha = np.repeat(alpha,nconst).reshape(-1,nconst)
    jetseta = jets[:,:,0]*np.cos(alpha) - jets[:,:,1]*np.sin(alpha)
    jetsphi = jets[:,:,0]*np.sin(alpha) + jets[:,:,1]*np.cos(alpha)

    #Calculate Centre of P and flip so always positive
    flip = np.sum((jetsphi * jets[:,:,-2]),axis=-1) < 0
    jetsphi = np.ma.where(np.repeat(flip,nconst).reshape(-1,nconst), -1*jetsphi, jetsphi)
    jets[:,:,:2] = np.ma.concatenate([jetseta.reshape(-1,nconst,1),jetsphi.reshape(-1,nconst,1)],axis=-1)

    return jets

