import numpy as np
from numpy import *
import math
import itertools
import operator

# data
K = 6       # K is # spatial orbitals, 2K # spin orbitals
N = 6       # N electrons

# find all combinations of spin orbitals to put electrons in
config = list(itertools.combinations(range(2*K), N))
config = np.array(config)
total_config = config.shape[0]
# you can either restrict the spin quantum number or not
spin_up = 3
spin_down = 3
index = []
for idx in range(total_config):
    # the even_th spin orbitals can be seen as spin-up orbitals
    if (len(np.where(config[idx, :]%2 == 0)[0]) == (spin_up * config.shape[1]) / (spin_up + spin_down)):
        index.append(idx)
print('index', index)
config = config[index, :]
electron_occupy = np.zeros((config.shape[0],2*K))
# electron_occupy includes all possible configurations
# "1" represents occupied
# "0" represents unoccupied
for i in range(electron_occupy.shape[0]):
    electron_occupy[i,config[i]] = 1

# count the inverse number
def Inverse_num(A):
    y=0
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            if A[i]>A[j]:
                y=y+1
    return y

# count the # of different spin orbitals between two configurations
def differ_num(state_a, stat_b):
    num = 0
    for i in range(2 * K):
        if(state_a[i] == 1) & (state_b[i] == 0):
            num+=1
    occupy = []   # the order of spin orbitals where electrons locate both in state a and b
    flag1 = []    # the order of spin orbitals where electron only in state a
    flag2 = []    # the order of spin orbotals where electron only in state b
    for i in range(m):
        if(state_a[i]==1)&(state_b[i]==0):
            flag1.append(i)
        if(state_a[i]==0)&(state_b[i]==1):
            flag2.append(i)
        if(state_a[i]==1)&(state_b[i]==1):
            occupy.append(i)
    # below code want to find the order of electrons, actually clumsy, you can try a smarter way
    # for example; state a: [1,1,0,1,0,1,1,0,1,0,0]
    #              state b: [0,1,0,0,1,1,1,0,1,0,0]
    # there is 1 electron switch,  
    # order of electrons in state a: [1,2,3,4,5,6]
    # order of electrons in state b: [3,1,2,4,5,6]
    rank_10=1
    rank_11=2
    rank_20=1
    rank_21=2
    if num==1 or num==2:
        for i in range(len(occupy)):
            if occupy[i]<flag1[0] :
                rank_10=rank_10+1
            if occupy[i]<flag2[0] :
                rank_20=rank_20+1
    if num==2:
        for i in range(len(occupy)):
            if occupy[i]<flag1[1]:
                rank_11=rank_11+1
            if occupy[i]<flag2[1]:
                rank_21=rank_21+1
    A = range(N) + 1
    B = range(N) + 1
    if num==2:
        A.remove(rank_10)
        A.remove(rank_11)
        B.remove(rank_20)
        B.remove(rank_21)
        A.insert(0,rank_11)
        A.insert(0,rank_10)
        B.insert(0,rank_21)
        B.insert(0,rank_20)
    if num==1:
        A.remove(rank_10)
        B.remove(rank_20)
        A.insert(0,rank_10)
        B.insert(0,rank_20)
    alpha=Inverse_num(A)
    beta=Inverse_num(B)
    alpha=alpha+beta
    return(alpha,num,flag1,flag2,occupy)

# import the itegral of molecular orbitals
a=np.load('h1e.npy')
b=np.load('h2e.npy')

# spin orbital integral

def h(i,j):
    if ((i-j)%2!=0):
        return 0
    else:
        y=a[i//2][j//2]
    return y
def J(i,j,k,l):
    y=0
    if (abs(i-k)%2!=0) or (abs(j-l)%2!=0):
        y=0
    else:
        y=b[i//2][k//2][j//2][l//2]
    return y

# introduce the Slater-Condon rule for matrix evaluation
# single electron integral and double electron intergral between interactions
def rule_1e(state_a, state_b):
    (alpha,num,flag1,flag2,occupy) = differ_num(state_a, state_b)
    M1 = 0
    if num==0:
        for i in occupy:
            M1 = M1+h(i,i)
    elif num==1:
        M1 =((-1)**alpha)*h(flag1[0],flag2[0])
    else:
        M1=0
    return M1
def rule_2e(state_a, state_b):
    M2 = 0
    (alpha,num,flag1,flag2,occupy) = differ_num(state_a,state_b)
    if num==0:
        for i in occupy:
            for j in occupy:
                M2 = M2+0.5*J(i,j,i,j)-0.5*J(i,j,j,i)
    elif num==1:
        for j in occupy:
            M2 = M2+(J(flag1[0],j,flag2[0],j)-J(flag1[0],j,j,flag2[0]))
        M2=((-1)**alpha)*M2
    elif num==2:
            M2 =((-1)**alpha)*(J(flag1[0],flag1[1],flag2[0],flag2[1])-J(flag1[0],flag1[1],flag2[1],flag2[0]))
    else:
        M2=0
    return M2

# form hamiltonian matrix'
Hamiltonian = np.zeros((len(electron_occupy),len(electron_occupy)))
print(shape(Hamiltonian))

for i in range(len(electron_occupy)):
    for j in range(len(electron_occupy)):
        Hamiltonian[i][j] = rule_1e(electron_occupy[i,:],electron_occupy[j,:])+rule_2e(electron_occupy[i,:],electron_occupy[j,:])

# direct diagonalization
w,v = np.linalg.eigh(Hamiltonian)
print(w)

