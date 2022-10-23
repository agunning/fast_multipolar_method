from sklearn.cluster import KMeans
import numpy as np
import scipy as sp
from tree_manager import *
from aux_functions import *
import matplotlib.pyplot as plt

def run_grav_algorithm(sources,sinks,values, tottolerance, weights=None, deg=2,attn=gaussian_attn, comparison_test=None):
    if weights is None:
        weights = np.ones(sources.shape[0])#np.exp((sources**2).sum(axis=1)/2)
    if isinstance(tottolerance,float):
        tottolerance = tottolerance*np.ones(sinks.shape[0])
    v=values.shape[1]
    d=sinks.shape[1]
    sources_tree=build_tree(sources,d,v,deg,weights,tree_dict={})
    sinks_tree=build_tree(sinks,d,v,deg,tree_dict={})
    errors_tree_sinks=max_tree(sinks_tree.keys(),keep_sum=False)
    errors_trees_sources={}
    for sink in sinks_tree.keys():    
        if sinks_tree[sink].no_leaves==0:
            errors_tree_sinks.set_max(sink,-tottolerance[sinks_tree[sink].labels].min())
        errors_trees_sources[sink]=max_tree([])
    totalerror=attn.tol_fun(sources_tree[()].rad,sinks_tree[()].rad,sources_tree[()].loc,sinks_tree[()].loc,deg)*weights.sum()
    errors_tree_sinks.set_add((),totalerror)
    errors_trees_sources[()].add_node(())
    errors_trees_sources[()].set_add((),totalerror)
    output=np.zeros((sinks.shape[0],values.shape[1]))
    t=0
    rel_errors=[]
    while errors_tree_sinks.node_max[()]>0:
        while errors_tree_sinks.node_max[()]>0:
            if t%1000==0:
                print(t, errors_tree_sinks.node_max[()])
                if comparison_test is None:
                    continue
                else:
                    t+=1
                    break
            t+=1
            worst_sl=errors_tree_sinks.get_max()
            
            adds=[errors_tree_sinks.add_cont[worst_sl[:i]] for i in range(len(worst_sl)+1)]
            sinkb= worst_sl[:np.argmax(adds)]
            #print(errors_tree_sinks.max_cont)
            #print(errors_tree_sinks.add_cont)
            sourceb = errors_trees_sources[sinkb].get_max()
            sourcen=sources_tree[sourceb]
            sinkn=sinks_tree[sinkb]
            errors_trees_sources[sinkb].set_max(sourceb,0)
            bf=errors_trees_sources[sinkb].bf[sourceb]
            #..
            #1:sinks are brute_forced
            #2:sources are brute forced
            if (((sinkn.rad>sourcen.rad)|(bf==2))&(bf!=1)):

                if sinkn.no_leaves==0:
                    newerror=attn.tol_fun(sourcen.rad,sinkn.rad,sourcen.loc,sinkn.loc,deg,1+bf)*sourcen.weights.sum()
                    #print(newerror)
                    if newerror<0:
                        print("negative error!")
                        errors_trees_sources[sinkb].set_max(sourceb,newerror)
                        errors_trees_sources[sinkb].bf[sourceb]+=1
                else:
                    errors_trees_sources[sinkb].set_max(sourceb,0)
                    errors_trees_sources[sinkb].bf[sourceb]=-1
                for i in range(sinks_tree[sinkb].no_leaves):
                    newsinkn=sinks_tree[sinkb+(i,)]
                    if newsinkn.nlabels>d:     
                        newbf=bf
                    else:
                        newbf=bf|1
                    newerror= attn.tol_fun(sourcen.rad,newsinkn.rad,sourcen.loc,newsinkn.loc,deg,newbf)*sourcen.weights.sum()*sourcen.weights.sum()
                    if newerror<0:
                        print("negative error!")
                    #print(errors_trees_sources[sinkb+(i,)].bf,i,sinkb,sourceb)
                    errors_trees_sources[sinkb+(i,)].add_node_fc(sourceb,max_cont=newerror,bf=newbf,sinkb= sinkb+(i,))
                    #print(errors_trees_sources[sinkb+(i,)].bf,i)
                    #print(errors_trees_sources[sinkb+(i,)].max_cont_sum)
                    errors_tree_sinks.set_add(sinkb+(i,),errors_trees_sources[sinkb+(i,)].max_cont_sum[()])
                errors_tree_sinks.set_add(sinkb,errors_trees_sources[sinkb].max_cont_sum[()])
                
            else:
                if sourcen.no_leaves==0:
                    newerror=attn.tol_fun(sourcen.rad,sinkn.rad,sourcen.loc,sinkn.loc,deg,2+bf)*sourcen.weights.sum()
                    if newerror<0:
                        print("negative error!")
                    errors_trees_sources[sinkb].set_max(sourceb,newerror)
                    errors_trees_sources[sinkb].bf[sourceb]+=2
                else:
                    errors_trees_sources[sinkb].set_max(sourceb,0)
                    errors_trees_sources[sinkb].bf[sourceb]=-1
                for j in range(sourcen.no_leaves):
                    newsourcen=sources_tree[sourceb+(j,)]
                    if newsourcen.nlabels>d:     
                        newbf=bf
                    else:
                        newbf=bf|1
                    newerror= attn.tol_fun(newsourcen.rad,sinkn.rad,newsourcen.loc,sinkn.loc,deg,newbf)*sourcen.weights.sum()                
                    if newerror<0:
                        print("negative error!")
                    errors_trees_sources[sinkb].add_node_fc(sourceb+(j,),max_cont=newerror,bf=newbf,sinkb=sinkb)
                errors_tree_sinks.set_add(sinkb,errors_trees_sources[sinkb].max_cont_sum[()])
        for sourceb in sorted(sources_tree.keys(),reverse=True):
            sourcen=sources_tree[sourceb]
            for i in range(0,deg+1):
                sourcen.deg[i]*=0
            
            if sourcen.no_leaves==0:
                sourcen.deg=build_moments(sources[sourcen.labels]-sourcen.loc,deg,weights[sourcen.labels],values[sourcen.labels])
            else:
                for j in range(sourcen.no_leaves):
                    nsourcen=sources_tree[sourceb+(j,)]
                    to_add=antishift(nsourcen.deg,sourcen.loc-nsourcen.loc,batchv=False)
                    for i in range(0,deg+1):
                        sourcen.deg[i]+=to_add[i]
        J=np.zeros(4)
        output*=0
        for sinkb in sorted(sinks_tree.keys()):
            sinkn=sinks_tree[sinkb]
            for i in range(0,deg+1):
                sinkn.deg[i]*=0
            for sourceb in sorted(errors_trees_sources[sinkb].nodes_list):
                bf=errors_trees_sources[sinkb].bf[sourceb]
                if bf == -1:
                    continue
                sourcen=sources_tree[sourceb]

                #print(bf,len(sinkn.labels),len(sourcen.labels))
                bf= errors_trees_sources[sinkb].bf[sourceb]
                wts=weights[sourcen.labels]
                vals=values[sourcen.labels]
                #bf=3
                if bf==0 and len(sourcen.labels)>d and len(sinkn.labels)>d:
                    J[0]+=len(sinkn.labels)*len(sourcen.labels)
                    for i in range(0,deg+1):
                        for j in range(0,deg+1):
                            sinkn.deg[i]+=attn.exp_fun(sourcen.deg[j],sourcen.loc,sinkn.loc,j,i,d,0)
                elif bf==1 and len(sinkn.labels)>d:
                    J[1]+=len(sinkn.labels)*len(sourcen.labels)
                    for j in range(0,deg+1):
                        output[sinkn.labels]+=attn.exp_fun(sourcen.deg[j],sourcen.loc,sinks[sinkn.labels],j,0,d,1)
                elif bf==2 and len(sourcen.labels)>d:
                    J[2]+=len(sinkn.labels)*len(sourcen.labels)
                    for i in range(0,deg+1):
                        sinkn.deg[i]+=attn.exp_fun(wts.reshape((-1,1))*vals,+sources[sourcen.labels],sinkn.loc,0,i,d,2)
                else:
                    J[3]+=len(sinkn.labels)*len(sourcen.labels)
                    output[sinkn.labels]+=attn.bf_fun(sources[sourcen.labels],sinks[sinkn.labels],values[sourcen.labels],weights[sourcen.labels])
            if sinkn.no_leaves==0:
                output[sinkn.labels]+=eval_powerseries(sinkn.deg,sinks[sinkn.labels]-sinkn.loc,deg)
            else:
                for i in range(sinkn.no_leaves):
                    nsinkn=sinks_tree[sinkb+(i,)]
                    shifted_deg = shift(sinkn.deg,nsinkn.loc-sinkn.loc,False)
                    for j in range(0,deg+1):
                        nsinkn.deg[j]+=shifted_deg[j]
        if not (comparison_test is None):
            rel_error=(np.abs(comparison_test-output)/tottolerance.reshape((-1,1))).mean()
            print("Time "+str(t)+", average Relative error is "+str(rel_error))
            rel_errors.append(rel_error)
            
        np.set_printoptions(suppress=False)
        print(J/J.sum())
    if (comparison_test is None):
        return output
    else:
        return output,rel_errors




def calc_denominator(sinks,sources,weights=None, sources_tree=None, reltolerance=0.5,calc = deg0est):
    if weights is None:
        if calc == deg0est:
            weights = np.exp((sources**2).sum(axis=1)/2)
        else:
            weights = np.ones(sources.shape[0])
    
    if sources_tree is None:
        sources_tree=build_tree(sources,weights,min_nodes=1,tree_dict={})
    sinks_tree=build_tree(sinks,min_nodes=1,tree_dict={})
    #print(sources_tree[()].rad,sinks_tree[()].rad,sources_tree[()].loc,sinks_tree[()].loc,weights.sum())
    totalest,totalerror=calc(sources_tree[()].rad,sinks_tree[()].rad,sources_tree[()].loc,sinks_tree[()].loc,weights.sum())
    #print(totalest,totalerror)
    sinks_ests=np.zeros(sinks.shape[0])
    sinks_errors=np.zeros(sinks.shape[0])
    sinks_rel_errors={}
    sinks_bp_ests={}
    sinks_bp_errors={}
    sinks_bp_sinkb={}
    sinks_errors_ub={}
    sinks_ests_ub={}
    for label in sinks_tree[()].labels:
        sinks_ests_ub[label]=totalest
        sinks_errors_ub[label]=totalerror
        sinks_ests[label]=totalest
        sinks_errors[label]=totalest
        sinks_rel_errors[label]=totalerror/totalest
        sinks_bp_ests[label]={():totalest}
        sinks_bp_errors[label]={():totalerror}
        sinks_bp_sinkb[label]={():()}
    t=0
    while max(sinks_rel_errors.values())>reltolerance:
        if t%10000==0:
            print(t, max(sinks_rel_errors.values()))
        t+=1
        u=max(sinks_rel_errors,key=sinks_rel_errors.get)       
        if len(sinks_bp_errors[u])==0:
            sinks_rel_errors[u]=0
        #print(sinks_rel_errors[u])
        sourceb=max(sinks_bp_errors[u],key=sinks_bp_errors[u].get)
        sinkb=sinks_bp_sinkb[u][sourceb]
        if sinks_tree[sinkb].rad>sources_tree[sourceb].rad:
            if (sinks_tree[sinkb].no_leaves)==0:           
                sinks_bp_errors[u][sourceb]=0
                sinks_errors[u]=sum(sinks_bp_errors[u].values())
                sinks_rel_errors[u]=sinks_errors[u]/sinks_ests[u]
                print("no leaves (sinkb)")
            for i in range(sinks_tree[sinkb].no_leaves):
                newest, newerror= calc(sources_tree[sourceb].rad,sinks_tree[sinkb+(i,)].rad,sources_tree[sourceb].loc,sinks_tree[sinkb+(i,)].loc,sources_tree[sourceb].weights.sum())
                for label in sinks_tree[sinkb+(i,)].labels:
                    if (sinks_errors[label]-sinks_bp_errors[label][sourceb]<.000001*sinks_errors_ub[label]): 
                        sinks_bp_errors[label][sourceb]=newerror
                        sinks_errors[label]=sum(sinks_bp_errors[label].values())
                        sinks_errors_ub[label]=sinks_errors[label]
                    else:
                        sinks_errors[label]+=(newerror-sinks_bp_errors[label][sourceb])
                        sinks_errors_ub[label]=max(sinks_errors[label],sinks_errors_ub[label])
                        sinks_bp_errors[label][sourceb]=newerror
                    if (sinks_ests[label]-sinks_bp_ests[label][sourceb]<.0001*sinks_ests_ub[label]):
                        sinks_bp_ests[label][sourceb]=newest
                        sinks_ests[label]=sum(sinks_bp_ests[label].values())
                        sinks_errors_ub[label]=sinks_errors[label]
                    else:
                        sinks_ests[label]+=newest-sinks_bp_ests[label][sourceb]
                        sinks_bp_ests[label][sourceb]=newest
                        sinks_ests_ub[label]=max(sinks_ests[label],sinks_ests_ub[label])
                    sinks_rel_errors[label]=sinks_errors[label]/sinks_ests[label]
                    sinks_bp_sinkb[label][sourceb]=sinkb+(i,)
        else:
            if (sources_tree[sourceb].no_leaves)==0:
                sinks_bp_errors[u][sourceb]=0
                sinks_errors[u]=sum(sinks_bp_errors[u].values())
                sinks_rel_errors[u]=sinks_errors[u]/sinks_ests[u]
                print(max(sinks_rel_errors.values()))
                print("no leaves (sourceb)")
                print(sources_tree[sourceb].rad,sinks_tree[sinkb].rad)
                print(sinks_errors_ub[u])
                print(sinks_errors[u])
                print(sinks_ests_ub[u])
                print(sinks_ests[u])
                continue
             
            #print(u,sinks_tree[sinkb].labels)
            for label in sinks_tree[sinkb].labels:
                #print(label==u)
                #print( sinks_bp_sinkb[u],sinks_bp_sinkb[label], sourceb)
                if (sinks_errors[label]-sinks_bp_errors[label][sourceb]<.000001*sinks_errors_ub[label]):
                    _=sinks_bp_errors[label].pop(sourceb,None)
                    sinks_errors[label]=sum(sinks_bp_errors[label].values())
                    sinks_errors_ub[label]=sinks_errors[label]
                else:
                    sinks_errors[label]-=sinks_bp_errors[label][sourceb]
                    _=sinks_bp_errors[label].pop(sourceb,None)
                    sinks_errors_ub[label]=max(sinks_errors[label],sinks_errors_ub[label])

                if (sinks_ests[label]-sinks_bp_ests[label][sourceb]<.000001*sinks_ests_ub[label]):
                    _=sinks_bp_ests[label].pop(sourceb,None)
                    sinks_ests[label]=sum(sinks_bp_ests[label].values())

                    sinks_ests_ub[label]=sinks_ests[label]
                else:
                    sinks_ests[label]-=sinks_bp_ests[label][sourceb]
                    _=sinks_bp_ests[label].pop(sourceb,None)
                    sinks_ests_ub[label]=max(sinks_ests[label],sinks_ests_ub[label])
                _=sinks_bp_sinkb[label].pop(sourceb,None)
            for j in range(sources_tree[sourceb].no_leaves):
                #print(sources_tree[sourceb].no_leaves)
                newest,newerror= calc(sources_tree[sourceb+(j,)].rad,sinks_tree[sinkb].rad,sources_tree[sourceb+(j,)].loc,sinks_tree[sinkb].loc,sources_tree[sourceb+(j,)].weights.sum())
                for label in sinks_tree[sinkb].labels:
                    sinks_errors[label]+=newerror
                    sinks_ests[label]+=newest
                    sinks_bp_errors[label][sourceb+(j,)]=newerror
                    sinks_bp_ests[label][sourceb+(j,)]=newest
                    sinks_bp_sinkb[label][sourceb+(j,)]=sinkb
            for label in sinks_tree[sinkb].labels:
                sinks_rel_errors[label]=sinks_errors[label]/sinks_ests[label]
    print(t)
    return sinks_ests,sinks_errors,sinks_rel_errors,sources_tree,sinks_tree
np.random.seed(0) #so we don't get new values every time we run this
bk=np.load('bert-keys-mini.npy')/2**1.5
bq=np.load('bert-queries-mini.npy')/2**1.5
v=np.random.normal(size=(2**16,4))

def test_error_decrease(k,values,eps=0.01):
    den=denom(k)
    bf_grav_output = bf_grav(bq[:k],bk[:k],values,np.exp((np.linalg.norm(bk[:k],axis=1)**2)/2))
    output,precision = run_grav_algorithm(bq[:k],bk[:k],values=values,tottolerance=eps*den,weights=np.exp((np.linalg.norm(bk[:k],axis=1)**2)/2),comparison_test=bf_grav_output)
    return output,precision
    
    

def denom(k):
    return bf_denominator(bq[:k],bk[:k],np.exp((np.linalg.norm(bk[:k],axis=1)**2)/2))
def num(k,expwt=True):
    if expwt:
        return bf_grav(bq[:k],bk[:k],np.exp((np.linalg.norm(bk[:k],axis=1)**2)/2),v[:k])
    else:
        return bf_grav(bq[:k],bk[:k],np.ones(k),v[:k])
def test2d(a=16,b=16,r1=.01,r2=.1,deg=2,dis=np.array([0,1]),rt=0.1):
    A=np.random.normal(0,r1,(a,2))+dis
    B=np.random.normal(0,r2,(b,2))
    W=np.exp((A**2).sum(axis=1)/2)
    #tt=rt*bf_denominator(A,B,W)
    tt=1e50*np.ones((b,1))
    val=np.ones((a,1))
    X = run_grav_algorithm(A,B,val,tt,deg=deg)[:,0]
    Y = bf_grav(A,B,val[:,0],W)
    print(X)
    print(Y)
    plt.scatter(A[:,0],A[:,1],c='black',marker='x')
    plt.scatter(B[:,0],B[:,1],c=X-Y)
    print(np.average((B**2).sum(axis=1)*(X-np.average(X))))
    print(np.average((B**2).sum(axis=1)*(Y-np.average(Y))))
    print(np.average(X-Y),np.std(X-Y))
    print(np.std(X))
    print(np.std(Y))
    print(np.mean(X))
    print(np.mean(Y))
    plt.show()
    plt.hist(X-Y)
    plt.show()
    return X,Y
