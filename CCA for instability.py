#fenbu
#e-xigema
#尝试百万级节点
#CCA论文
#沃特斯坦距离
#fenbu
#e-xigema
#尝试百万级节点
#CCA论文
#沃特斯坦距离
import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.cross_decomposition import CCA
import string
from sklearn.preprocessing import StandardScaler

from numpy import dot
import scipy.linalg as la
import matplotlib.pyplot as plt

import numpy as np
import ast
from sklearn import preprocessing
from scipy.stats import pearsonr
from scipy import linalg

if __name__ == "__main__":

    dicdata = {'cora':2708,'mi':2277,'gr':5242,'ca':12008,'p2p':8846,'wiki':7115,'web':121422,'bra':131,'ka':34,'citeseer':3312,'pubmed2':19717}
    sall = 'cora'
    salg = ''
    snum = '1'
    sssnum = 1  #h0+hsssnum
    #strname = 'wpqvec/'230902pminq1walklen15
    strname = 'corawalklength/emb3/length50'
    sdata = 'datanei/'+sall
    weidu = 64
    num1 = 0
    num2 = 0
    jq = 0
    xx1 = ""
    xx2 = ""
    x1 = []
    x2 = []
    tmp = []
    hg = 0
    hgg = []
    varhg = []
    wg = []

    newlocalx1 = []
    newlocalx2 = []  #

    #'''
    dict1 = {}
    file1 = open(sdata + "/nei1.txt", "r") #读取每个结点的一阶邻居
    row = file1.readlines()
    iddd = []
    for line in row:
        str = list(line.split(' '))
        dict1[eval(str[0])] = eval(str[1])
        iddd.append(eval(str[0]))
    fiddd = open(strname + "iddd.txt",'w')
    print('[', end='', file=fiddd)
    for idd in iddd:
        print(idd,end=',',file=fiddd)
    print(']', end='', file=fiddd)

    '''
    dict2 = {}
    file2 = open(sdata + "/nei2.txt", "r") #读取每个结点的2阶邻居
    row = file2.readlines()
    for line in row:
        str = list(line.split(' '))
        dict2[eval(str[0])] = eval(str[1])

    dict3 = {}
    file3 = open(sdata + "/nei3.txt", "r") #读取每个结点的3阶邻居
    row = file3.readlines()
    for line in row:
        str = list(line.split(' '))
        dict3[eval(str[0])] = eval(str[1])
    '''
    nodenum = dicdata[sall] #2708 2277 gr5242 12008 p2p8846 7115 121422
    f = open(strname + '/macro'+salg+sall+snum+'.txt', 'w')
    ff1 = open(strname + '/wg'+snum+'0.txt', "w")#weiguan
    with open(strname + "/h2.txt") as get:      #基准矩阵
         xx2 = get.read()
         xx2 = eval(xx2)
         get.close()

    for i in xx2:
        x2 = i

    np.set_printoptions(suppress=True)                 #

    for i in range(sssnum,sssnum+1):
        jq = jq+1
        x1 = []
        sna = strname + "/h"+'%d'% i +".txt"
        with open(sna) as get1:  # 计算矩阵
            xx1 = get1.read()
            xx1 = eval(xx1)
            get1.close()
        for ii in xx1:
            x1 = ii

        x1 = np.array(x1)     #测试矩阵中的一个
        x2 = np.array(x2)     #基准矩阵

        wgx1 = x1
        wgx2 = x2


        normalizer_data = preprocessing.Normalizer().fit_transform(x1)
        normalizer_data = preprocessing.Normalizer().fit_transform(x2)

        cca = CCA(n_components=weidu)
        cca.fit(x1, x2)
        #print(cca.x_weights_[0])
        xn = []
        yn = []
        zn = []
        for ccaw in range(0,weidu):
            kn = 0
            k = 0
            for ccawi in cca.x_loadings_[ccaw]:
                k = k+ccawi*ccawi
                kn =kn+1
            xn.append(k/kn)

        for ccaw in range(0,weidu):
            kn = 0
            k = 0
            for ccawi in cca.y_loadings_[ccaw]:
                k = k+ccawi*ccawi
                kn =kn+1
            yn.append(k/kn)
        # 降维操作
        # print(X)
        X_train_r, Y_train_r = cca.transform(x1, x2)       #将两个矩阵降维
        newlocalx1 = X_train_r.T #降维后的矩阵
        newlocalx2 = Y_train_r.T
        summm = 0
        fp = open('pvaluegraphsagecora.txt','w')
        for tri in range(0, weidu):
            cf = np.corrcoef(X_train_r[:, tri], Y_train_r[:, tri])[0, 1]
            stat, p = pearsonr(X_train_r[:, tri], Y_train_r[:, tri])
            #print(p)
            print(p,file = fp)
            if p <= 0.05:
                summm = summm+1
                hg = hg+cf
        fp.close()
        hg = hg/summm
        hgg.append(hg)



    ans = []

    dord={}

    #'''
    forder = open(strname + "/orderlist.txt",'r')  #按embeddings读取节点顺序
    row = forder.readlines()
    numorder = 0
    #for i in row:
        #dord[eval(i)]=numorder
        #numorder = numorder+1
    for i in iddd:
        dord[i] = numorder
        numorder = numorder+1

    print('[',end='',file = ff1)
    for nodei in range(0,nodenum):
        wgcf = np.corrcoef(newlocalx1[:, nodei], newlocalx2[:, nodei])[0, 1]
        wg.append(wgcf)
        print(wgcf,end=',',file = ff1)
    print(']', end='', file=ff1)

    '''
    file4 = open(sdata + "/list.txt", "r")
    row = file4.readlines()
    it = 0
    fans1 = open(strname + "/wg"+snum+"1.txt", "w")
    print('[',end='',file = fans1)
    for line in row:
        it = it + 1
        ll = dict1[eval(line)]
        l1 = []
        l2 = []
        w1ans = 0
        ll.append(eval(line))
        for ill in ll:
            numill = dord[ill]
            w1ans = w1ans+wg[numill]
        w1ans = w1ans/len(ll)
        print(w1ans,end=',',file = fans1)
    print(']',file = fans1)

    it = 0
    fans2 = open(strname + "/wg"+snum+"2.txt", "w")
    print('[',end='',file = fans2)
    for line in row:
        it = it + 1
        ll = dict2[eval(line)]
        l1 = []
        l2 = []
        w1ans = 0
        ll.append(eval(line))
        for ill in ll:
            numill = dord[ill]
            w1ans = w1ans+wg[numill]
        w1ans = w1ans/len(ll)
        print(w1ans,end=',',file = fans2)
    print(']',file = fans2)

    it = 0
    fans3 = open(strname + "/wg"+snum+"3.txt", "w")
    print('[',end='',file = fans3)
    for line in row:
        it = it + 1
        ll = dict3[eval(line)]
        l1 = []
        l2 = []
        w1ans = 0
        ll.append(eval(line))
        for ill in ll:
            numill = dord[ill]
            w1ans = w1ans+wg[numill]
        w1ans = w1ans/len(ll)
        print(w1ans,end=',',file = fans3)
    print(']',file = fans3)
    '''
    varhgans = np.var(varhg)
    print(np.var(hgg))
    print(np.mean(hgg))
    print(varhgans, file=f)
    #print(varhgans)
    print(hg,file=f)
    print(ans,file=f)

    f.close()
'''
0.6443866277646806
0.6498730199917383

'''
'''
0.5706680866330857
0.5874891734033554

'''
'''
0.4915485319777104

'''