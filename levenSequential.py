# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
import time
''' ########### read two strings from the mat file ############'''
label_1 = loadmat('G:\My Drive\Projects\Event Detection\dataset\LabellerIdx_7_PrIdx_1_TrIdx_1.mat')
label_2 = loadmat('G:\My Drive\Projects\Event Detection\dataset\LabellerIdx_8_PrIdx_1_TrIdx_1.mat')
data1 = label_1['TrialData']
data2 = label_2['TrialData']
labels_1 = data1['Labels']
labels_2 = data2['Labels']
temp = labels_1[0,0]
temp2 = labels_2[0,0]
temp = temp[6]
temp2 = temp2[7]
labeller1 = temp[0][0]
labeller2 = temp2[0][0]
labeller2 = labeller2[0:len(labeller1)]
N = 10000
str1 = labeller1[:N]
str2 = labeller2[:N]


H = np.zeros(shape=(N+1,N+1),dtype=np.int32)
H[0,:] = np.arange(N+1)
H[:,0] = np.arange(N+1)
#Ht = np.reshape(H,-1)
start = time.time()
for SLICE in range(1,N*2):
    if SLICE<N:
        z = 1
    else:
        z = SLICE - N +1
    for j in range(z,SLICE-z+2):
        row = j
        column = SLICE-j +1
        if row==0 or column==0:
            continue
        if str1[row-1]==str2[column-1]:
            score = 0
        else:
            score = 1
        H[row,column] = min(H[row-1,column-1]+score,H[row-1,column]+1,H[row,column-1]+1)

duration = time.time()-start
score_final = H[-1,-1]
print('final score is %d'% score_final)
print('Sequential duration is %s' % duration)    