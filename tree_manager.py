import numpy as np
class max_tree:
    def __init__(self,nodes_list,keep_sum=True):
        self.no_children={}
        self.add_cont={}
        self.max_cont={}
        self.node_max={}
        self.largest_child={}
        self.keep_sum=keep_sum
        self.max_cont_sum={}
        self.nodes_list=nodes_list
        self.bf={}
        if(keep_sum):
            self.max_cont_sum={}
        for node in nodes_list:
            self.no_children[node]=0
            self.bf[node]=0
            self.largest_child[node]=-1
            self.add_cont[node]=0
            self.max_cont[node]=-np.inf
            self.node_max[node]=-np.inf
            if self.keep_sum:
                self.max_cont_sum[node]=-np.inf
        for node in nodes_list:
            if len(node)>0:
                self.no_children[node[:-1]]+=1
    def _recompute(self,node):
        no_children=self.no_children[node]
        options=np.zeros(no_children+1)
        for i in range(no_children):
            options[i]=self.node_max[node+(i,)]
        options[no_children]=self.max_cont[node]
        u=np.argmax(options)
        if u==no_children:
            self.node_max[node]=self.add_cont[node]+self.max_cont[node]
            self.largest_child[node]=-1
        else:
            self.node_max[node]=self.add_cont[node]+self.node_max[node+(u,)]
            self.largest_child[node]=u
    def _prop(self,node):
        child=node
        while(len(child)>0):
            parent=child[:-1]
            if self.node_max[child]+self.add_cont[parent]>self.node_max[parent]:
                self.node_max[parent]=self.node_max[child]+self.add_cont[parent]
                #print(self.largest_child)
                #print(child[-1])
                self.largest_child[parent]=child[-1]
            elif(child[-1]==self.largest_child[parent]):
                self._recompute(parent)
            else:
                break
            child=parent
    def _recompute_sums(self,node):
        if self.keep_sum:
            parent=node+(0,)
            while len(parent)>0:
                parent=parent[:-1]
                self.max_cont_sum[parent]=self.max_cont[parent]+sum(self.max_cont_sum[parent+(i,)] for i in range(self.no_children[parent]))
    def set_add(self,node,add_cont):
        if node not in self.add_cont.keys():
            print("node does not exist")
            return -1
        #delta=add_cont-self.add_cont[node]
        #self.node_max[node]+=delta :thonking emoji:
        if self.largest_child[node]==-1:
            self.node_max[node]=self.max_cont[node]+add_cont
        else:
            i=self.largest_child[node]
            self.node_max[node]=self.node_max[node+(i,)]+add_cont
        self.add_cont[node]=add_cont
        self._prop(node)
    def add_node(self,node,add_cont=0,max_cont=-np.inf,bf=0):
        if (node !=()):
            if (not (node[:-1] in  self.add_cont.keys())):
                print("cannot add node")
                return -1
            if (node[-1]!=0 and (node[:-1]+(node[-1]-1,) not in self.add_cont.keys())):
                print("cannot add node")
                return -1
        if node in self.add_cont.keys():
            print("cannot add node")
            return -1
        self.nodes_list.append(node)
        self.add_cont[node]=add_cont
        self.node_max[node]=max_cont+add_cont
        self.max_cont[node]=max_cont
        if self.keep_sum:
            self.max_cont_sum[node]=max_cont
        self.no_children[node]=0
        if node!=():
            self.no_children[node[:-1]]+=1
        self.largest_child[node]=-1
        self.bf[node]=bf
        self._prop(node)
        self._recompute_sums(node)
    def add_node_fc(self,node,add_cont=0,max_cont=0,df_add_cont=0,df_max_cont=0,bf=0,sinkb=()):
        if node in self.add_cont.keys():
            if self.bf[node]!=-1:
                print('warning: node already added')
                print(sinkb,node,add_cont)
                print(self.node_max)
                print(self.max_cont)
                print(self.max_cont_sum)
            self.set_add(node,add_cont)
            self.set_max(node,max_cont)
            self.bf[node]=0
        else:
            nstack=[]
            cnode=node
            while not(cnode in self.add_cont.keys()):
                nstack.append(cnode)
                if cnode == ():
                    break
                if cnode[-1]==0:
                    cnode =cnode[:-1]
                else:
                    cnode = cnode[:-1]+(cnode[-1]-1,)
            while len(nstack)>1:
                self.add_node(nstack.pop(),df_add_cont,df_max_cont,-1)
            self.add_node(node,add_cont,max_cont,bf)
        
    def set_max(self,node,max_cont):
        if node not in self.max_cont.keys():
            print("node does not exist")
            return -1
        delta=max_cont-self.max_cont[node]
        self.max_cont[node]=max_cont
        self._recompute_sums(node)
        if delta>=0:
            if self.node_max[node]<max_cont+self.add_cont[node]:
                self.node_max[node]=max_cont+self.add_cont[node]
                self.largest_child[node]=-1
                self._prop(node)
        else:
            if self.largest_child[node]==-1:
                self._recompute(node)
                self._prop(node)
    def get_max(self):
        node=()
        while self.largest_child[node]!=-1:
            node = node+(self.largest_child[node],)
        return node
    def get_all_values(self,val_dict=None,node=(),add=0):
        if val_dict is None:
            val_dict={}
        val_dict[node]=add+self.node_max[node]
        for i in range(self.no_children[node]):
            self.get_all_values(val_dict,node+(i,),add+self.add_cont[node])
        return val_dict
