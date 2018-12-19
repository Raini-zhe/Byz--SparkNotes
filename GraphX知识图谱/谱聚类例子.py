import numpy as np
import networkx as nx
import scipy.linalg as llg
from Queue import PriorityQueue
import matplotlib.pylab as plt
import heapq as hp
from sklearn.cluster import k_means

# fake data from multivariate normal distribution 随机产生多维高斯分布点
S = np.random.multivariate_normal([1,1], [[0.5,0],[0,0.7]],100)
T = np.random.multivariate_normal([-1,-1], [[0.3,0],[0,0.8]],100)
R = np.random.multivariate_normal([-1,0], [[0.4,0],[0,0.5]],100)
# >>> A1_mean = [1, 1]
# >>> A1_cov = [[2, .99], [1, 1]]
# >>> A1 = np.random.multivariate_normal(A1_mean, A1_cov, 10) #依据指定的均值和协方差生成数据
# >>> A1
# array([[-1.72475813,     0.33681971],
#          [ 0.78643798,      0.76700529],
#          [ 0.61538183,      -0.75786666],...
Data = np.vstack([S,T,R])
# >>> a = np.array([1, 2, 3])
# >>> b = np.array([2, 3, 4])
# >>> np.vstack((a,b))
# array([[1, 2, 3],
#        [2, 3, 4]])
plt.subplot(1,2,1)
plt.scatter(S.T[0],S.T[1],c='r')
plt.scatter(T.T[0],T.T[1],c='b')
plt.scatter(R.T[0],R.T[1],c='y')

# calc k-nearest neighbors ，计算前k小值
def min_k(i,k):
    pq = []
    for j in range(len(Data)):
        if i == j:
            continue
        if len(pq) < k: # 如果堆元素小于维度k，就追加,
            hp.heappush( pq,(1/np.linalg.norm(Data[i]-Data[j]), j) )
            #heapq.heappush(heap,item)  #heap为定义堆，item为增加的元素;
            # ,Data[i]是一个向量：
            #>>> np.linalg.norm( np.array([ 0.43987372, -0.72650078]) - np.array([ -1.20064895e+00,  -1.09746593e+00]) )
            #   1.6819422621774065 ->得到向量的L2范数。用1相除目的是做大小转化，大值变小值
        else:
            hp.heappushpop( pq, (1/np.linalg.norm(Data[i]-Data[j]), j) )
            #heapq.heappushpop(heap, item)  #向堆里插入一项，并弹出返回heap里最小值的项。若item是个元组（i,j）,则函数会只比较i的大小，j忽略
            #：问题
            #   ：我们要求最小值为何还要弹出最小值，因为～ 用一除以一个最大值即为最小值，实际弹出的是原始的最大值
            #   >>> 1/1.6819422621774065
            #    0.5945507301216282
            #   >>> 1/2.6819422621774065
            #    0.37286410453449625
    return pq

# calc laplacian
L = np.zeros((len(Data),len(Data)))
for i in range(len(Data)):
    for (v,j) in min_k(i, 3):  #计算稀疏化后的图的laplacian矩阵，计算其前K小特征值对应的特征向量作为矩阵H的列
        L[i,j] = -v
        L[j,i] = -v #对称？
L = L + np.diag(-np.sum(L,0)) #diag对角线放值，如[2, 2, 5]
# >>> a=np.sum([[0,1,2],[2,1,3]],axis=0)
# >>> a
# array([2, 2, 5])

# kmean
(lam, vec) = llg.eigh(L)
# llg.eigh解决复杂的Hermitian或真实对称矩阵的普通或广义特征值问题。
# 找到矩阵a的特征值w和可选的特征向量v ，其中 b是正定：
A = vec[:,0:3]
kmean = k_means(A,3)

plt.subplot(1,2,2)
plt.scatter(Data.T[0],Data.T[1],c=['r' if v==0 else 'b' if v==1 else 'y' for v in kmean[1]])
plt.show()



np.linalg.norm(np.array([ 0.43987372, -0.72650078])-np.array([ -1.20064895e+00,  -1.09746593e+00]))
