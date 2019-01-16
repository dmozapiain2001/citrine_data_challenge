# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:13:21 2018

@author: vadit
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 20:58:06 2018

@author: vadit
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.autograd import Variable
import scipy.io as sio
import numpy as np
from timeit import default_timer
class damage_nn(nn.Module):
    def __init__(self,pc,l1_size,l2_size,l3_size,l4_size,l5_size,l6_size,op4):
        super(damage_nn,self).__init__()
        self.fc1=nn.Linear(pc,l1_size)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(l1_size,l2_size)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(l2_size,l3_size)
        self.relu3=nn.ReLU()
        self.fc4=nn.Linear(l3_size,l4_size)
        self.relu4=nn.ReLU()
#        self.do4=nn.Dropout(p=0.2)
        self.fc5=nn.Linear(l4_size,l5_size)
        self.relu5=nn.ReLU()
#        self.do5=nn.Dropout(p=0.2)
        self.fc6=nn.Linear(l5_size,l6_size)
        self.relu6=nn.ReLU()
#        self.do6=nn.Dropout(p=0.2)
        self.fc7=nn.Linear(l6_size,op4)
    def forward(self,x):
        out=self.fc1(x)
        out=self.relu1(out)
        out=self.fc2(out)
        out=self.relu2(out)
        out=self.fc3(out)
        out=self.relu3(out)
        out=self.fc4(out)
        out=self.relu4(out)
#        out=self.do4(out)
        out=self.fc5(out)
        out=self.relu5(out)
#        out=self.do5(out)
        out=self.fc6(out)
        out=self.relu6(out)
#        out=self.do6(out)
        out=self.fc7(out)
        return out
lenpc=50
pcr=sio.loadmat('pcr.mat')
pcr=pcr['pcr']
pcr0=pcr.shape[0]
#tset=np.random.randint(1,776,size=70)
tset=np.arange(297,374)
nte=len(tset)
ntr=pcr0-nte-1
trset=np.setdiff1d(np.arange(pcr0),tset)
print(pcr0,nte,ntr)
N,pc,l1_size,l2_size,l3_size,l4_size,l5_size, l6_size,op4=ntr,lenpc,70,40,30,20,10,5,2
tps_pc=sio.loadmat('tps_pc.mat')
tps_pc=tps_pc['tps_pc']


#pc_i=np.zeros((ntr,lenpc))
#pc_i[0:215,:]=pcr[0:215,0:lenpc]
#pc_i[215:694,:]=pcr[297:776,0:lenpc]
pc_i=pcr[trset,0:lenpc]
pc_test=pcr[tset,0:lenpc]
pc_cv=np.zeros(lenpc)
pc_cv[:]=pcr[pcr0-1,0:lenpc]
sbparams=sio.loadmat('sbparams.mat')
sbparams=sbparams['sbparams']
dvals=np.zeros((ntr,2))
dvals_cv=np.zeros(2)
#sbparams.shape
#dvals[0:215,:]=sbparams[0:215,:]
#dvals[215:694,:]=sbparams[297:776,:]
dvals=sbparams[trset,:]
#dval_test=sbparams[215,297,:]
dval_test=sbparams[tset,:]
dvals_cv[:]=sbparams[pcr0-1,:]
#pc_i=np.zeros((270,20))
#pc_i[0:30,:]=pcr[0:30,:]
#pc_i[30:270,:]=pcr[60:300,:]
#pest=sio.loadmat('paramEsts.mat')
#pest=pest['paramEsts']
#davg=sio.loadmat('d_avg_calc.mat')
##davg
#davg=davg['d_avg']
#davg=np.reshape(davg,(270,1))
#davg
#sbparams=sio.loadmat('sbparams.mat')
#sbparams=sbparams['sbparams']
#dvals=np.zeros((270,op4))
#dvals[:,0:op4]=sbparams
#dvals[:,3]=np.reshape(davg,(270,))
print(dvals.shape)
def nae_calc(gt,pred,ns,nd):
    gtsum=np.zeros(nd)
    predsum=np.zeros(nd)
    for jj in range(nd):
        gtsum[jj]=np.sum(gt[:,jj])
        predsum[jj]=np.sum(pred[:,jj])
    nmae=np.zeros(nd)
    for jj in range(nd):
        for ii in range(ns):
            nmae[jj]=nmae[jj]+np.abs((gt[ii,jj]-pred[ii,jj])/(gtsum[jj]))
    
    nae=np.zeros((ns,nd))
    for i in range(ns):
        for j in range(nd):
            nae[i,j]=np.abs((pred[i,j]-gt[i,j])/gt[i,j])
    return nae,nmae

dvals=torch.from_numpy(dvals)

dvals.requires_grad_(True)
pc_i=torch.from_numpy(pc_i)

pc_i.requires_grad_(True)
pc_i=pc_i.float()
pc_cv=torch.from_numpy(pc_cv)
pc_cv.requires_grad_(True)
pc_cv=pc_cv.float()
dvals_cv=torch.from_numpy(dvals_cv)
pc_test=torch.from_numpy(pc_test).float()
dvals_cv.requires_grad_(True)
#eqn=np.loadtxt('class-c-1.txt',delimiter=',',dtype=int)
#pc_test=np.zeros((len(eqn),20))
#for jj in range(len(eqn)):
#    pc_test[jj,:]=tps_pc[99+eqn[jj],:]
#for i in range(10):
#    fnm="class_comma_"+str('%04d' %(i+1,))+".txt"
#    ary=np.loadtxt(fnm,delimiter=',')
#    ary=ary.astype(int)
#    pcc=tps_pc[i*100:(i+1)*100]
#    pcr[i*30:(i+1)*30]=pcc[ary[:]-1,:]

#model3=damage_nn(pc,l1_size,l2_size,l3_size,l6_size, op4)
model3=damage_nn(pc,l1_size,l2_size,l3_size,l4_size,l5_size,l6_size, op4)
model3=torch.load('model3-temp.dat')
#model2=torch.load('model.dat')
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model3.parameters(),lr=1e-3)
start=default_timer()
step=0
loss_cv0=0
import matplotlib.pyplot as plt
plt.figure()
for t in range(40000):
#    step=default_timer()-start
    dpred=model3(pc_i).double()
    loss=criterion(dpred[:,0].double(),dvals[:,0].double())
    if(t%100==0):
        step=default_timer()-step
        print(t,loss.item(),step)    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dpred_cv=model3(pc_cv).double()
    loss_cv=criterion(dpred_cv.double(),dvals_cv.double())
    plt.plot(t,loss_cv.item(),'r*')
#    if(t!=0 and loss_cv0<loss_cv.item()):
#        break
    loss_cv0=loss_cv.item()
    
    if (t%100==0):
        print(t,"CV error -- ",loss_cv.item(),step)
        dtest=model3(pc_test).double()
        dpred_n=dpred.detach().numpy()
        dvals_n=dvals.detach().numpy()
        #dval_test=sio.loadmat('sbparams_test.mat')
        #dval_test=dval_test['sbparams_test']
        #dval_test=dval_test[:,0:op4]
        dtest_n=dtest.detach().numpy()
        #for i in range(270):
        #    for j in range(4):
        #        nae[i,j]=np.abs((dpred_n[i,j]-dvals_n[i,j])/dvals_n[i,j])*100.0
        nae_test,nmae_test=nae_calc(dval_test,dtest_n,nte,op4)
        nae_tr,nmae_tr=nae_calc(dvals_n,dpred_n,ntr,op4)
        print(nmae_test,nmae_tr)
#        if(nmae_test[0]>0.15 or nmae_tr[0]>0.15):
#            break
#    if(nmae_test[0]>0.15 or nmae_tr[0]>0.15):
#        break
endtime=default_timer()-start


