from sklearn.cluster import KMeans
import numpy as np
import scipy as sp
import math

class leaf_node:
    def __init__(self, ident, loc, rad,labels,weights,d,v,deg):
        self.ident=ident
        self.rad=rad
        self.loc=loc
        self.no_leaves=0
        self.labels=labels
        self.nlabels=len(labels)
        self.weights=weights
        self.deg=[np.zeros((v,)+(d,)*i) for i in range(0,deg+1)]
def build_tree(locations,d,v,deg, weights=None,labels=None,min_nodes=None,branch_factor=8, tree_dict=None,current_tag=()):
    #print(len(tree_dict.keys()))
    if tree_dict is None:
        tree_dict={}
    if labels is None:
        labels = np.array(list(range(locations.shape[0])),dtype=np.int64)
    if weights is None:
        weights=np.array([1.]*locations.shape[0])
    if min_nodes is None:
        min_nodes=locations.shape[1]
    if len(labels)==1:
        rad=0
        mu=locations[0]
    else:     
        mu = np.average(locations,axis=0, weights=weights)   
        rad=max(np.linalg.norm(locations-mu,axis=1))
    tree_dict[current_tag]=leaf_node(current_tag,mu,rad,labels,weights,d,v,deg)
    if weights.shape[0]<=min_nodes:
         return tree_dict
    kmeans = KMeans(n_clusters=min(branch_factor,weights.shape[0])).fit(locations,sample_weight=weights)
    #print(kmeans.labels_==0)
    indices= np.argsort(kmeans.labels_)
    ###print(np.cumsum(np.bincount(kmeans.labels_)))
    slices = np.concatenate((np.array([0]),np.cumsum(np.bincount(kmeans.labels_))))
    
    for i in range(len(slices)-1):
        slindices=indices[slices[i]:slices[i+1]]
        _=build_tree(locations[slindices],d,v,deg,weights[slindices],labels[slindices],min_nodes,branch_factor,tree_dict,current_tag+(i,))
        tree_dict[current_tag].no_leaves+=1
    return tree_dict

def part_symmetrize(tensor,k):
    A=tensor/k
    B=A.copy()
    for j in range(2,k+1):
        B+=A.swapaxes(-1,-j)
    return B 

def symmetrize(tensor,n):
    #symmetrizes the last n axes of a tensor
    #we love ourselves some Jucys-Murphy elements
    A=tensor/np.math.factorial(n)
    for i in range(2,n+1):
        B=A.copy()
        for j in range(1,i):
            B+=A.swapaxes(-i,-j)
        A=B
    return A


    

def gaussianexpand(A,u1,u2,m,n,d,bf=0,gaussian=True):
    #a is a rank M symmetric tensor, want th
    v=u2-u1
    ans=0
    if bf==1:
        A=np.expand_dims(A,axis=0)
    for i in range(min(m,n)+1):
        for j in range((m-i)//2+1):
            T=A.copy()
            for l in range(m-i-2*j):
                if bf!=0:
                    T=np.einsum('ij...k,ik -> ij...', T,v)
                else:
                    T=T@v
            for l in range(0,j):
                T=np.trace(T,axis1=-1,axis2=-2)
            S=T.copy()
            #print(T.shape)
            for k in range((n-i)//2+1):
                R=S.copy()
                for l in range(n-i-2*k):
                    if bf!=0:
                        R=np.einsum('ij...,ik -> ij...k', R,-v)
                    else:
                        R=np.tensordot(R,-v,axes=0)
                lam = (-1)**(i+j+k)
                lam*=np.prod(range(m,m-i,-1))*np.prod(range(n,n-i,-1))/(np.math.factorial(i))
                lam*=sp.special.binom(m-i,2*j)*sp.special.binom(n-i,2*k)
                lam*=np.prod(range(1,2*j+1,2))*np.prod(range(1,2*k+1,2))
                lam/=(math.factorial(n)*math.factorial(m))
                if bf==2:
                    ans+=lam*R
                else:
                    ans+=lam*R
                if k!=(n-i)//2:
                    S=np.tensordot(S,np.identity(d),axes=0)
    if bf==0:
        ans*=np.exp(-(v**2).sum()/2)
    elif bf==1:
        ans = np.einsum('i...,i->i...',ans,np.exp(-(v**2).sum(axis=1)/2))
    else:
        ans = np.einsum('i...,i->...',ans,np.exp(-(v**2).sum(axis=1)/2))
    return symmetrize(ans,n)

def dtprod_expand(A,u,v,m,n,d,bf=0,gaussian=True):
    #a is a rank M symmetric tensor, want th
    
    ans=0
    if bf==1:
        A=np.expand_dims(A,axis=0)
    for i in range(min(m,n)+1):
        T=A.copy()
        for l in range(m-i):
            if bf!=0:
                T=np.einsum('ij...k,ik -> ij...', T,v)
            else:
                T=T@v
            #print(T.shape)
        for l in range(n-i):
            if bf!=0:
                T=np.einsum('ij...,ik -> ij...k', R,u)
            else:
                T=np.tensordot(R,u,axes=0)
            lam=np.prod(range(m,m-i,-1))*np.prod(range(n,n-i,-1))/(np.math.factorial(i))
            lam/=(math.factorial(n)*math.factorial(m))
            if bf==2:
                ans+=lam*R
            else:
                ans+=lam*R
    if bf==0:
        ans*=np.exp((u*v).sum())
    elif bf==1:
        ans = np.einsum('i...,i->i...',ans,np.exp((u*v).sum(axis=1)/2))
    else:
        ans = np.einsum('i...,i->...',ans,np.exp((u*v).sum(axis=1)/2))
    return symmetrize(ans,n)


def shift(coeffs,v,batchv=True):
    #shifts a power series by v
    n=len(coeffs)-1
    d=coeffs[-1].shape[-1]
    ans = []
    clist=[]
    for i in range(n,-1,-1):
        for j,elt in enumerate(clist):
            if batchv:
                clist[j]=np.einsum('i...k,ik->i...',elt,v)
            else:
                clist[j]=elt @v 
        clist.append(coeffs[i])
        ncoeff=np.zeros_like(coeffs[i])
        for j,elt in enumerate(clist):
            t=n-j
            ncoeff+=sp.special.binom(t,i)*(-1)**(t-i)*elt
        ans.append(ncoeff)
    return list(reversed(ans))


def antishift(coeffs,v,batchv=True):
    n=len(coeffs)-1
    d=coeffs[-1].shape[-1]
    ans=[]
    clist=[]
    for i in range(0,n+1):
        for j,elt in enumerate(clist):
            if batchv:
                clist[j]=part_symmetrize(np.einsum('i...,ik->i...k',elt,v),i)
            else:
                clist[j]=part_symmetrize(np.tensordot(elt,v,axes=0),i)
        clist.append(coeffs[i])
        if batchv:
            ncoeff=np.zeros(coeffs[i].shape[1:])
        else:
            ncoeff=np.zeros_like(coeffs[i])
        for j, elt in enumerate(clist):
            if batchv:
                ncoeff+=sp.special.binom(i,j)*elt.sum(axis=0)
            else:
                ncoeff+=sp.special.binom(i,j)*elt
        ans.append(ncoeff)
    return ans

def build_moments(v,n,wts,vals):
    X=wts.reshape((-1,1))*vals
    ans = [X.sum(axis=0)]
    for i in range(1,n+1):
        X=np.einsum('i...,ik->i...k',X,v)
        ans.append(X.sum(axis=0))
    return ans

def eval_powerseries(A,v,deg):
    rsum = np.expand_dims(A[-1],axis=0)
    for i in range(deg-1,-1,-1):
        rsum=np.einsum('ij...k,ik->ij...',rsum,v)
        rsum+=A[i]
    return rsum
        
    
    
    #transpose of shift
    
def absHermite(n,x):
    err=0
    for i in range(0,2*((n//2)+1),2):
        err+=np.prod(range(2*i+1,n+1))*np.prod(range(1,2*i+1,2))/np.prod(range(1,n-2*i+1))*x**(n-2*i)
    return err



def tolerance(rad1,rad2,loc1,loc2,d=2,bf=0):
    #bf=1: sinks are atomized
    #bf=2:sources are atomized
    #bf=3:
    dist=np.linalg.norm(loc1-loc2)
    if bf==3:
        return 0
    elif bf==2:
        return rad1**(d+1)*np.math.factorial(d+1)**(-1)*absHermite(d+1,dist+rad1+rad2)*np.exp(-(max(0,dist-rad1-rad2)**2/2))
    elif bf==1:
        return rad2**(d+1)*np.math.factorial(d+1)**(-1)*absHermite(d+1,dist+rad1+rad2)*np.exp(-(max(0,dist-rad1-rad2)**2/2))
    else:
        
        err1=0
        if True:
            for i in range(0,d+1):
                err1+=rad1**(d+1)*rad2**i*np.math.factorial(d+1)**(-1)*np.math.factorial(i)**(-1)*absHermite(d+1+i,dist+rad1)*np.exp(-(max(0,dist-rad1)**2/2))
                err1+=rad2**(d+1)*rad1**i*np.math.factorial(d+1)**(-1)*np.math.factorial(i)**(-1)*absHermite(d+1+i,dist+rad2)*np.exp(-(max(0,dist-rad2)**2/2))
            err1+=(rad1*rad2)**(d+1)*np.math.factorial(d+1)**(-2)*absHermite(d+1+i,dist+rad2+rad1)*np.exp(-(max(0,dist-rad2-rad2)**2/2))
        if True:
            err2=(rad1**(d+1)+rad2**d+1)*np.math.factorial(d+1)**(-1)*np.math.factorial(i)**(-1)*absHermite(d+1+i,dist+rad2)*np.exp(-(max(0,dist-rad1-rad2)**2/2))
        #probably empirically the first one of these is better but eh
        return min(err1,err2)




def dtprod_tolerance(rad1,rad2,loc1,loc2,deg=2,bf=0):
    err=0
    d1=np.linalg.norm(loc1)
    d2=np.linalg.norm(loc2)
    cdot = (loc1*loc2).sum()
    maxexp =cdot+d1*rad2+d2*rad1+rad1*rad2
    
    c3=np.exp(np.sum(loc1*loc2)+d2*rad1+d1*rad1+rad*rad1)
    if bf ==3:
        return 0
    if bf==2:
        return rad1**(d+1)*np.math.factorial(d+1)**(-1)*(loc2+rad2)**(d+1)*np.exp(maxexp)
    if bf == 1:
        return rad2**(d+1)*np.math.factorial(d+1)**(-1)*(loc1+rad1)**(d+1)*np.exp(maxexp)
    else:
        e1=np.exp(cdot+rad1*d2)
        e2=np.exp(cdot+rad2*d1)
        for i in range(0,d+1):
            erc1=0
            erc2=0
            for j in range(0,i+1):
                erc1+=(cdot+rad1*d2)**j*(d2)**(d-j)*(d1+rad1)**(i-j)
                erc2+=(cdot+rad2*d1)**j*(d1)**(d-j)*(d2+rad2)**(i-j)
            err+=erc1*e1*np.math.factorial(d+1)**(-1)*np.math.factorial(i)**(-1)*rad1**(d+1)*rad2**i
            err+=erc1*e1*np.math.factorial(d+1)**(-1)*np.math.factorial(i)**(-1)*rad1**i*rad2**(d+1)
        erc=0
        for j in range(0,d+1):
            erc+=maxexp**j*(d2+rad2)**(d-j)*(d1+rad1)**(d-j)
        err+=erc*np.exp(maxexp)*np.math.factorial(d+1)**(-1)*np.math.factorial(i)**(-1)*rad1**(d+1)*rad2**(d+1)
    return err




def deg0dtproduct(rad1,rad2,loc1,loc2,wt):
    #print(rad1,rad2,np.sum(loc1*loc2))
    c=np.exp(np.sum(loc1*loc2))*wt
    a=np.exp(np.sum(loc1*loc2)+np.linalg.norm(loc1)*rad2+np.linalg.norm(loc2)*rad1+rad1*rad2)
    b=np.exp(np.sum(loc1*loc2)-np.linalg.norm(loc1)*rad2-np.linalg.norm(loc2)*rad1-rad1*rad2)
    return c, min((a-b)/2,a*rad1*(np.linalg.norm(loc2)+rad2))


             
    
def deg0est(rad1,rad2,loc1,loc2,wt):
    dist=np.linalg.norm(loc1-loc2)
    a=np.exp(-(max(0,dist-rad1-rad2))**2/2)*wt
    b=np.exp(-(dist+rad1+rad2)**2/2)*wt
    c=np.exp(-dist**2/2)*wt
    if(rad1==0) & (rad2==0):
        return c, 0
    return c,min((a-b)/2,rad1*a*(dist+rad1+rad2))





def bf_denominator(sources,sinks,weights):
    (m,d)=sources.shape
    (n,_)=sinks.shape
    return (np.exp(-((sources.reshape((1,m,d))-sinks.reshape(n,1,d))**2).sum(axis=-1)/2)*weights).sum(axis=1)

def bf_grav(sources,sinks,weights,values):
    (m,d)=sources.shape
    (n,_)=sinks.shape
    X=np.exp(-((sources.reshape((1,m,d))-sinks.reshape((n,1,d)))**2).sum(axis=-1)/2)
    return (X*weights)@values



def bf_dotprod_denominator(sources,sinks,weights=1):
    (m,d)=sources.shape
    (n,_)=sinks.shape
    return np.exp(sources.reshape(1,m,d)*sinks.reshape(n,1,d)).sum(axis= (1,2))


    
def bf_dotprod_grav(sources,sinks,values,weights=1):
    (m,d)=sources.shape
    (n,_)=sinks.shape
    return (np.exp(sources.reshape(1,m,d)*sinks.reshape(n,1,d)).sum(axis= -1)*weights)@values

    


#there's probably a much more principled way to do htis

class attn_fun:
    def __init__(self,exp_fun,tol_fun,bf_fun):
        self.exp_fun=exp_fun
        self.tol_fun=tol_fun
        self.bf_fun=bf_fun

gaussian_attn=attn_fun(
    exp_fun=gaussianexpand,
    tol_fun=tolerance,
    bf_fun=bf_grav)
dtprod_attn=attn_fun(
    exp_fun=dtprod_expand,
    tol_fun=dtprod_tolerance,
    bf_fun=bf_dotprod_grav)
