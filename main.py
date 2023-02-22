import torch
from net import *
import argparse
import csv
import datetime
import shutil
from smiles2vector import *
import scipy.io as io
import networkx as nx
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, DataLoader

side_effect_label = r"E:\python\机器学习\side_effect_label_750.mat"
input_dim = 109
cuda_name='cuda:0'
model=GAT3().to(device=cuda_name)
model.load_state_dict(torch.load(r'E:\python\机器学习\net_params.pth'),strict=False)
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
print('Device: ', device)
#loader=
DF=False
not_FC=False
knn=5
pca=False
metric='cosine'
# 生成副作用的graph信息
frequencyMat=np.loadtxt(r'E:\python\机器学习\frequencyMat.csv',delimiter=',',dtype='int')
#print(frequencyMat)
frequencyMat = frequencyMat.T
if pca:
    pca_ = PCA(n_components=256)
    similarity_pca = pca_.fit_transform(frequencyMat)
    print('PCA 信息保留比例： ')
    print(sum(pca_.explained_variance_ratio_))
    A = kneighbors_graph(similarity_pca, knn, mode='connectivity', metric=metric, include_self=False)
else:
    A = kneighbors_graph(frequencyMat, knn, mode='connectivity', metric=metric, include_self=False)
G = nx.from_numpy_matrix(A.todense())
edges = []
for (u, v) in G.edges():
    edges.append([u, v])
    edges.append([v, u])

edges = np.array(edges).T
edges = torch.tensor(edges, dtype=torch.long)

node_label = io.loadmat(side_effect_label)['node_label']
feat = torch.tensor(node_label, dtype=torch.float)
sideEffectsGraph = Data(x=feat, edge_index=edges)
def predict(model, device,x,edge_index,batch, sideEffectsGraph, DF, not_FC):
    model.eval()
    torch.cuda.manual_seed(42)
    print('Make prediction for {} samples...'.format(1))
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        pred, _, _ = model(x,edge_index,batch, sideEffectsGraph, DF, not_FC)
    return pred

#举例子
smile_graph = convert2graph(['C[N+](C)(C)CC(=O)[O-]'])
x=torch.FloatTensor(smile_graph['C[N+](C)(C)CC(=O)[O-]'][1])
edge_index=torch.LongTensor(smile_graph['C[N+](C)(C)CC(=O)[O-]'][2])
batch=[0 for i in range(len(x))]
batch=torch.LongTensor(batch)
a=predict(model,device,x,edge_index,batch,sideEffectsGraph,DF,not_FC)

res=[a[0][i].item() for i in range(994)]


path=r"E:\QQ文件下载\Supplementary Data 1.txt"
fr=open(path,'r')
all_lines=fr.readlines()
dataset=[]
for line in all_lines:
    line=line.strip().split('\t')
    dataset.append(line)
#print(dataset)
df=pd.DataFrame(dataset[1:],columns=['drug','sideeffect','rate'])
data=list(set(list(df['sideeffect'])))
data.sort()

df=pd.DataFrame(res,index=data)
print(df)