from pycuda.autoinit import context
import pycuda.driver as cuda
import numpy as np
from scipy.io import loadmat
from pycuda.compiler import SourceModule
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
width = 10000
labeller1 = labeller1[:width]
labeller2 = labeller2[:width]

'''############### copy the strings to the managed memory ##################'''
#width = len(labeller1)
start = time.time()
str1 = cuda.managed_zeros(shape=width, dtype= np.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
str2 = cuda.managed_zeros(shape=width, dtype= np.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
H = cuda.managed_zeros(shape=(width+1,width+1),dtype=np.int32,mem_flags=cuda.mem_attach_flags.GLOBAL)
H[0,:] = np.arange(width+1)
H[:,0] = np.arange(width+1)
Ht = cuda.managed_zeros(shape=(width+1)*(width+1),dtype=np.int32,mem_flags=cuda.mem_attach_flags.GLOBAL) 
for row in range(width+1):
    for column in range(width+1):
        index = row*(width+1)+column
        Ht[index] = H[row,column]
str1[:] = labeller1[:]
str2[:] = labeller2[:]

#print(str1[:50])
#print(str2[:50])

mod = SourceModule("""
__global__ void lenvenshtein(int *str1, int *str2, int *Ht, int N, int z, int slice, int sliceSize)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if(ID >= sliceSize)
        {return;}
    //str1[blockIdx.x * blockDim.x + threadIdx.x] *= 2;
    //str2[blockIdx.x * blockDim.x + threadIdx.x] *= 2;
    int startIndex = 0;
    if(z==1)
    {
         startIndex = slice+1;
    }
    else
    {
         startIndex = N * z;
    }
    int index = startIndex + (ID+1) *(N-1);
    int row = index / N;
    int collumn = index % N;
    if(row==0||collumn==0)
        return;
    int score;
    if(str1[row-1]==str2[collumn-1])
        score = 0;
    else
        score = 1;
    int min = Ht[(row-1)*N+collumn-1]+score;
    if(Ht[(row-1)*N+collumn]+1<min)
        {min = Ht[(row-1)*N+collumn]+1;}
    if(Ht[row*N+collumn-1]+1<min)
        {min = Ht[row*N+collumn-1]+1; }
    Ht[index] = min;
}
""")
lenvenshtein = mod.get_function("lenvenshtein")
N = np.int32(width)
block_size = 512
start2 = time.time()
for SLICE in range(1,N*2):
    if SLICE<width:
        z = 1
        sliceSize = SLICE
    else:
        z = SLICE - width +1
        sliceSize = SLICE - 2*z + 2 
    lenvenshtein(str1,str2,Ht,N+1,np.int32(z),np.int32(SLICE),np.int32(sliceSize),grid=(sliceSize//block_size+1,1), block=(min(block_size,sliceSize),1,1))
    context.synchronize()

duration = time.time()-start
duration2 = time.time()-start2
H_result = np.reshape(Ht,(width+1,width+1))
score_final = H_result[-1,-1]
print('GPU computation duration: %s'% duration2)
print('GPU total duration: %s'% duration)
print('Final score is: %d' % score_final)

#print(str1[:50])
#print(str2[:50])