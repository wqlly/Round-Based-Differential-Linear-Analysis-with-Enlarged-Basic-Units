from __future__ import annotations
import math
import itertools
import numpy as np

Sbox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
       0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
#
P = [0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,
    4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,
    8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,
    12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63]
INVP = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,
       1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,
       2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,
       3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]

ROW_GROUPS = ((0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15))
COL_GROUPS = ((0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15))

def dot(a,x,N):
    z=a&x;r=0
    for i in range(N):r^=(z>>i)&1
    return r

def fwt(v):
    n=len(v);h=1
    while h<n:
        for i in range(0,n,2*h):
            for j in range(i,i+h):
                x=v[j];y=v[j+h]
                v[j]=x+y;v[j+h]=x-y
        h<<=1

def genDDT(S,N):
    s=1<<N
    d=np.zeros((s,s),dtype=np.float64)
    for u in range(s):
        for x in range(s):d[S[x]^S[x^u],u]+=1
    return d/s

def genLAT(S,N):
    s=1<<N
    lat=np.zeros((s,s),dtype=np.float64)
    for b in range(s):
        v=np.zeros(s,dtype=np.float64)
        for x in range(s):v[x]=1 if dot(b,S[x],N)==0 else -1
        fwt(v)
        lat[b]=v
    return lat/s

def passInvP(X):
    y=np.zeros(64,dtype=X.dtype)
    for i in range(64):y[INVP[i]]=X[i]
    return y

def log2_abs(x):
    return math.log2(abs(x)) if x!=0 else -float('inf')

def tuple_to_index(t):
    return ((t[0]*16+t[1])*16+t[2])*16+t[3]

def index_to_tuple(idx):
    return (idx>>12)&15,(idx>>8)&15,(idx>>4)&15,idx&15

def cell_nibble(U,c):
    return 8*U[4*c]+4*U[4*c+1]+2*U[4*c+2]+U[4*c+3]

def init_super(X):
    #print(X)
    G=np.empty((4,16,16,16,16),dtype=np.float64)
    for g,grp in enumerate(ROW_GROUPS):
        t=np.multiply.outer(X[grp[0]],X[grp[1]])
        t=np.multiply.outer(t,X[grp[2]])
        t=np.multiply.outer(t,X[grp[3]])
        G[g]=t.reshape(16,16,16,16)
    #print(G)
    return G

def apply_axis(T,M,axis):
    y=np.tensordot(M,T,axes=([1],[axis]))
    return np.moveaxis(y,0,axis)

def apply_sbox(T,M):
    for ax in range(4):T=apply_axis(T,M,ax)
    return T

def p_row_col_pre():
    pre=np.empty(16**4,dtype=np.uint32)
    for i in range(16**4):
        #print(i)
        v=index_to_tuple(i)
        #print(v)
        u=[0,0,0,0]
        for q in range(4):
            m=v[q]
            for c in range(4):
                if (m>>(3-c))&1:
                    u[c]|=1<<(3-q)
        pre[i]=tuple_to_index(u)
        #print(hex(pre[i]))
    #print(pre[:])
    return pre
#p_row_col_pre()
def inter_pre():
    allp=[]
    for r in range(4):
        pre=np.empty((16**4,4),dtype=np.uint32)
        for i in range(16**4):
            v=index_to_tuple(i)
            u=[[0]*4 for _ in range(4)]
            for c in range(4):
                m=v[c]
                for s in range(4):
                    if (m>>(3-s))&1:
                        u[s][c]|=1<<(3-r)
            for s in range(4):
                pre[i,s]=tuple_to_index(u[s])
        allp.append(pre)
    #print(allp[0])
    return allp   

def apply_p_row(T,pre):
    return T.reshape(-1)[pre].reshape(16,16,16,16)



def apply_inter_p(G,pre_all):
    f=[g.reshape(-1) for g in G]
    out=np.empty_like(G)
    for r in range(4):
        p=pre_all[r]
        #print(pre_all[r])
        out[r]=(f[0][p[:,0]]*f[1][p[:,1]]*f[2][p[:,2]]*f[3][p[:,3]]).reshape(16,16,16,16)
        #print(out[r])
    return out
pre1=p_row_col_pre()
pre2=inter_pre()
def propagate(X,M,n):
    G=init_super(X)    
    orient='row'
    for _ in range(n):
        if orient=='row':
            for g in range(4):G[g]=apply_p_row(G[g],pre1)
            for g in range(4):G[g]=apply_sbox(G[g],M)
            orient='col'
        else:
            G=apply_inter_p(G,pre2)
            for g in range(4):G[g]=apply_sbox(G[g],M)
            orient='row'
    return G,orient

def eval_mask(G,orien,U):
    v=1.0
    grps=ROW_GROUPS if orien=='row' else COL_GROUPS
    for g,grp in enumerate(grps):
        t=tuple(cell_nibble(U,c) for c in grp)
        v*=G[g][t]
    return v

def final(G,orien,M,Mask):
    l=len(Mask)
    xx=np.zeros(1<<(4*l),dtype=np.float64)
    for v in range(1<<(4*l)):
        V=np.zeros(64,dtype=np.int8)
        for j in range(l):
            v0=(v>>(4*j))&15
            for k in range(4):V[4*Mask[j][0]+k]=(v0>>(3-k))&1
        U=passInvP(V)
        xx[v]=eval_mask(G,orien,U)
    mat=M
    for _ in range(l-1):
        mat=np.kron(mat,M)
    return mat@xx

def getBias_Opt4_Superbox(ROUND,LAT,Diff,Mask):
    DDT=genDDT(Sbox,4)
    M=LAT**2
    COR=0.0
    OD=[list(range(16)) for _ in Diff]
    for comb in itertools.product(*OD):
        ok=True
        for i in range(len(Diff)):
            if DDT[comb[i],Diff[i][1]]==0:ok=False;break
        if not ok:continue
        X=np.zeros((16,16),dtype=np.float64)
        
        for i in range(len(Diff)):X[Diff[i][0],comb[i]]=1.0
        for i in range(16):
            if np.sum(X[i])==0:X[i,0]=1.0
        for i in range(16):fwt(X[i])
        #print(X)
        G,ori=propagate(X,M,ROUND-2)
        res=final(G,ori,M,Mask)
        ccc=1.0
        for i in range(len(Diff)):ccc*=DDT[comb[i],Diff[i][1]]
        COR+=ccc*res
    return COR

if __name__=="__main__":
    LAT=genLAT(Sbox,4)
    tests=[(14,[[5,13]],[[10,11]],11)]
    for r,d,m,i in tests:
        print(f"ROUND {r}")
        res=getBias_Opt4_Superbox(r,LAT,d,m)[i]
        print(f"Result: {res}")
        print(f"log2: {log2_abs(res)}")
    fin=0
    for A in range(0,16):
       for a in range(1,16):
           for B in range(0,16):
               for b in range(1,16):
                   tests=[(11,[[A,a]],[[B,b]],b)]
                   for r,d,m,i in tests:
                       res=getBias_Opt4_Superbox(r,LAT,d,m)[i]
                       if abs(res)>abs(fin):
                           fin=res
                           print(f"input/output:{A} {a} / {B} {b}")
                           print(f"Result: {res}")
                           print(f"log2: {log2_abs(res)}")
    