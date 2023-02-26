import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import streamlit as st
import argparse
import os
import base64
import csv
import time
import datetime
import shutil
from net import *
from smiles2vector import *
import scipy.io as io
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import kneighbors_graph
from PIL import Image
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
cuda_name='cuda:0'
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import plotly.graph_objs as go

path=r"\Supplementary Data 1.txt"
pic_path=r"17593290.png"
model_path=r'net_params.pth'
frequence_path=r'frequencyMat.csv'
side_effect_path=r'side_effect_label_750.mat'
raw_frequency_path=r'raw_frequency_750.mat'

frequence=['none','非常罕见','罕见','不频繁','频繁','非常频繁']
#===================================直接在这里修改路径==========================================================


#_______________________________________________________翻译逻辑_________________________________________________________#

from translate import Translator
import requests
from rdkit import Chem

class LanguageTrans():
    def __init__(self, mode):
        self.mode = mode
        if self.mode == "E2C":
            self.translator = Translator(from_lang="english", to_lang="chinese")
        if self.mode == "C2E":
            self.translator = Translator(from_lang="chinese", to_lang="english")
    def trans(self, word):
        translation = self.translator.translate(word)
        return translation
# 中译英

def translate_name(chinese):
    '''
    返回药物英文名字
    :param chinese:药物中文名
    :return: 药物英文名
    '''
    translator = LanguageTrans("C2E")

    word = translator.trans(chinese)

    return word

def find_Molecular_formula(Eng_name):
    drug_name = Eng_name

    # PubChem API 的基本 URL
    base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'

    # 发送 HTTP GET 请求并获取响应
    response = requests.get(base_url + drug_name + '/property/MolecularFormula/JSON')

    # 输出响应内容
    return response.json()['PropertyTable']['Properties'][0]['MolecularFormula']

def get_Smiles(Eng_name):
    name = Eng_name
    api_key = 'your_api_key'
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON'
    response = requests.get(url, headers={'API_Key': api_key})

    # 解析响应
    if response.status_code == 200:
        data = response.json()
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        return smiles
    else:
        return None
#_______________________________________________________翻译逻辑_________________________________________________________________


#-------------------------------------------------------创建标题——————————————————————————————————————————————————————————————————
st.set_page_config(page_title = 'CUG-GAT药物副作用预测',page_icon = '🕵️‍♀️',layout = 'wide',initial_sidebar_state='expanded')

image_path = "机器学习/17593290.png"
absolute_path = os.path.abspath(image_path)


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('17593290.png')




st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)
st.write("<style>div.row-widget.stRadio > div > label{display:inline-block;padding-left:10px;padding-right:10px;}</style>", unsafe_allow_html=True)


st.write("<style>h1 {color: black;}</style>", unsafe_allow_html=True)#title
st.write("<style>p {color: black;}</style>", unsafe_allow_html=True)#text,
st.write("<style>div.stAlert > div > div > div.stAlert-text {color: blue;}</style>", unsafe_allow_html=True)#info
st.write("<style>h2 {color: white;}</style>", unsafe_allow_html=True)#header
st.write("<style>p2 {color: black;}</style>", unsafe_allow_html=True)#button
st.write("<style>div.stAlert > div > div > div.stAlert-text {color: black;}</style>", unsafe_allow_html=True)


st.title('CUG-GAT药物副作用预测')

st.sidebar.expander('')
st.sidebar.write('用户输入 ⬇️')
Top_K=st.sidebar.slider(label='选择 Top K副作用',min_value=1,max_value=25,value=(1,5))[1]

#---------------------------------------------------创建标题-------------------------------------------------------------


#-------------------------------------------------函数定义逻辑------------------------------------------------------------
#@st.cache
def load_model():
    side_effect_label = r"side_effect_label_750.mat"
    input_dim = 109
    cuda_name = 'cuda:0'
    model = GAT3().to(device=device)
    model.load_state_dict(torch.load(model_path), strict=False)
    return model

#@st.cache
def load_frequencyMat():
    frequencyMat = pd.read_csv(frequence_path, delimiter=',', dtype='float').values
    frequencyMat = frequencyMat.T
    return frequencyMat

#@st.cache
def load_node_label():
    side_effect_label = side_effect_path
    node_label = io.loadmat(side_effect_label)['node_label']
    return node_label

def search_by_name():
    # 输入药物名
    medecine_name = st.text_input('请在下方文本栏输入药物名称：', value='点击这里输入')

    try:
        Eng_name = translate_name(medecine_name)
    except Exception as e:
        Eng_name = None

    if Eng_name == None:
        Smiles = None
    else:
        Smiles = get_Smiles(Eng_name)

    # 构造API请求,点击确定获取药物分子式并进行预测
    if st.button('开始预测'):

        if Smiles != None:
            state = st.text('开始进行药物解析...')
            st.write('药物Smile分子式为:', Smiles)
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(50):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'解析进度:{i * 2 + 2}%')
                bar.progress(i * 2 + 1)
                time.sleep(0.1)
            # ----------------------------------------预测药物代码-----------------------
            # 转换为Smiles形式
            smile_graph = convert2graph([Smiles])
            x = torch.FloatTensor(smile_graph[Smiles][1])
            edge_index = torch.LongTensor(smile_graph[Smiles][2])
            batch = [0 for i in range(len(x))]
            batch = torch.LongTensor(batch)
            prob = predict(model, device, x, edge_index, batch, sideEffectsGraph, DF, not_FC)
            temp_result = [round(prob[0][i].item()) for i in range(994)]

            result=[frequence[x] for x in temp_result]

            dataframe=pd.DataFrame(temp_result,index=data,columns=['频率评分'])





            # ----------------------------------------预测药物代码-----------------------

            # 显示进度

            # dataframe排序
            dataframe.sort_values(by='频率评分', inplace=True, ascending=False)

            result=list(dataframe['频率评分'])

            class_=[frequence[score] for score in result]

            dataframe2=pd.DataFrame(class_,index=dataframe.index,columns=['频率评分'])
            # 加载完成
            st.success('药物解析...完成!', icon="✅")

            col1, col2 = st.columns(2)

            with col1:
                draw_topK(Top_K, dataframe)

                st.info(
                    '本软件旨在作用于科研药物研究，查询副作用只是作为参考，若您是患者，请到专业医疗机构咨询，切勿依赖本APP',
                    icon="ℹ️")
                st.info('药物分子数据来自于开源网站:https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/。',
                        icon="ℹ️")
            with col2:
                pic = draw_chaimcal(Smiles)

                st.header('显示药物分子图')

                st.image(pic, caption=Smiles)

                st.header('显示药物得分表')
                st.dataframe(dataframe2.style.highlight_max(axis=0))
            #draw_comparation(prob.cpu().detach())


        else:

            st.warning('未查询到药物,这可能是由于输入药物未收录或您的网络不畅。请检查网络或输入内容后重试', icon="⚠️")

def search_by_Chemical():
    chamical_name = st.text_input('请在这里输入药物分子式 ', value='点击这里输入')
    if st.button('开始预测'):
        if chamical_name != None:
            state = st.text('正在解析药物...')
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(50):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'解析进度:{i * 2 + 2}%')
                bar.progress(i * 2 + 1)
                time.sleep(0.1)
            # ----------------------------------------预测药物代码-----------------------
            # 转换为Smiles形式
            smile_graph = convert2graph([chamical_name])
            x = torch.FloatTensor(smile_graph[chamical_name][1])
            edge_index = torch.LongTensor(smile_graph[chamical_name][2])
            batch = [0 for i in range(len(x))]
            batch = torch.LongTensor(batch)
            prob = predict(model, device, x, edge_index, batch, sideEffectsGraph, DF, not_FC)

            temp_result = [round(prob[0][i].item()) for i in range(994)]

            dataframe = pd.DataFrame(temp_result, index=data, columns=['频率评分'])

            # ----------------------------------------预测药物代码-----------------------
            # 显示进度

            # dataframe排序
            dataframe.sort_values(by='频率评分', inplace=True, ascending=False)

            result=list(dataframe['频率评分'])

            class_=[frequence[score] for score in result]

            dataframe2=pd.DataFrame(class_,index=dataframe.index,columns=['频率评分'])


            # 加载完成
            st.success('药物解析...完成!', icon="✅")
            col1, col2 = st.columns(2)
            with col1:
                draw_topK(Top_K, dataframe)



                st.info(
                    '本软件旨在作用于科研药物研究，查询副作用只是作为参考，若您是患者，请到专业医疗机构咨询，切勿依赖本APP',
                    icon="ℹ️")
                st.info('药物分子数据来自于开源网站:https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/。',
                        icon="ℹ️")

            with col2:
                pic = draw_chaimcal(chamical_name)

                st.header('显示药物分子图')

                st.image(pic, caption=chamical_name)
                st.header('显示药物得分表')
                st.dataframe(dataframe2.style.highlight_max(axis=0))

        else:

            st.warning('未查询到药物，这可能是由于输入药物未收录或您的网络不畅。请检查网络或输入内容后重试',icon="⚠️")

def seach_side_effect_with_name():
    # 输入药物名
    medecine_name = st.text_input('请在下方文本栏输入药物名称：', value='清空这里输入')
    side_effect_name=st.text_input('在下方输入副作用名:',value='清空这里输入')

    try:
        Eng_name = translate_name(medecine_name)
    except Exception as e:
        Eng_name = None

    if Eng_name == None:
        Smiles = None
    else:
        Smiles = get_Smiles(Eng_name)
    if st.button('确定'):
        if Smiles != None:
            state = st.text('开始进行药物解析...')
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(50):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'解析进度:{i * 2 + 2}%')
                bar.progress(i * 2 + 1)
                time.sleep(0.1)
            # ----------------------------------------预测药物代码-----------------------
            # 转换为Smiles形式
            smile_graph = convert2graph([Smiles])
            x = torch.FloatTensor(smile_graph[Smiles][1])
            edge_index = torch.LongTensor(smile_graph[Smiles][2])
            batch = [0 for i in range(len(x))]
            batch = torch.LongTensor(batch)
            prob = predict(model, device, x, edge_index, batch, sideEffectsGraph, DF, not_FC)
            temp_result = [round(prob[0][i].item()) for i in range(994)]

            result = [frequence[x] for x in temp_result]
            dataframe = pd.DataFrame(result, index=data, columns=['频率评分'])
            # ----------------------------------------预测药物代码-----------------------

            # 显示进度

            # dataframe排序
            dataframe.sort_values(by='频率评分', inplace=True, ascending=False)

            # 加载完成
            st.success('药物解析...完成!', icon="✅")


            st.write('查询成功，副作用频率评分为：',dataframe.loc[side_effect_name,'频率评分'])

            st.info('本软件旨在作用于科研药物研究，查询副作用只是作为参考，若您是患者，请到专业医疗机构咨询，切勿依赖本APP',
                    icon="ℹ️")
            st.info('药物分子数据来自于开源网站:https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/。', icon="ℹ️")

        else:

            st.warning('未查询到药物,这可能是由于输入药物未收录或您的网络不畅。请检查网络或输入内容后重试', icon="⚠️")

def seach_side_effect_with_chemical():
    # 输入药物名
    medecine_name = st.text_input('请在下方文本栏输入药物分子式：', value='清空这里输入')
    side_effect_name = st.text_input('在下方输入副总用名:', value='清空这里输入')
    if st.button('确定'):
        Smiles = get_Smiles(medecine_name)

        if Smiles != None:
            state = st.text('开始进行药物解析...')
            st.write('药物Smile分子式为:', Smiles)
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(50):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'解析进度:{i * 2 + 2}%')
                bar.progress(i * 2 + 1)
                time.sleep(0.1)
            # ----------------------------------------预测药物代码-----------------------
            # 转换为Smiles形式
            smile_graph = convert2graph([Smiles])
            x = torch.FloatTensor(smile_graph[Smiles][1])
            edge_index = torch.LongTensor(smile_graph[Smiles][2])
            batch = [0 for i in range(len(x))]
            batch = torch.LongTensor(batch)
            prob = predict(model, device, x, edge_index, batch, sideEffectsGraph, DF, not_FC)
            temp_result = [round(prob[0][i].item()) for i in range(994)]

            result = [frequence[x] for x in temp_result]
            dataframe = pd.DataFrame(result, index=data, columns=['频率评分'])
            #dataframe = pd.DataFrame(temp_result, index=data, columns=['频率评分'])
            # ----------------------------------------预测药物代码-----------------------

            # 显示进度

            # dataframe排序
            dataframe.sort_values(by='频率评分', inplace=True, ascending=False)

            # 加载完成
            st.success('药物解析...完成!', icon="✅")

            st.write('查询成功，副作用频率评分为：', dataframe.loc[side_effect_name, '频率评分'])

        else:

            st.warning('未查询到药物,这可能是由于输入药物未收录或您的网络不畅。请检查网络或输入内容后重试', icon="⚠️")

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

def draw_chaimcal(str_smi):
    mol = Chem.MolFromSmiles(str_smi)
    return 

def draw_topK(k,dataframe):
    sub_df=dataframe.iloc[:k]
    sub_data=np.array(sub_df)
    # y=[sub_data[i][0] for i in range(len(sub_data)) ]
    # fig,ax=plt.subplots()
    # ax.bar(sub_df.index.tolist(),y)
    # plt.xticks(rotation=90)
    # st.pyplot(fig)
    #st.bar_chart(sub_df)
    st.header('副作用评分柱状图')
    fig = go.Figure(
        data=[go.Bar(x=sub_df.index, y=sub_df['频率评分'], name='副作用1评分')],
        layout=go.Layout(
            title='',
            xaxis=dict(title='样本编号'),
            yaxis=dict(title='评分'),
        )
    )
    st.plotly_chart(fig)

def draw_comparation(model_res):

    dataa=io.loadmat(raw_frequency_path)

    a=model_res

    a = list(np.array(a))

    truth_data = dataa['R']
    t = truth_data[8]
    index = np.array(np.where(t == 0))
    index_x = np.array(np.where(t != 0))
    for i in range(len(index[0])):
        a[0][index[0][i]] = 0

    p = []
    tru = []
    effect = []
    # estradiol
    for i in range(15):
        print(data[index_x[0][i]], round(a[0][index_x[0][i]]), t[index_x[0][i]])
        p.append(round(a[0][index_x[0][i]]))
        tru.append(t[index_x[0][i]])
        effect.append(data[index_x[0][i]])
    size = 15

    x = np.arange(size)

    # 有a/b两种类型的数据，n设置为2
    total_width, n = 0.6, 2
    # 每种类型的柱状图宽度
    width = total_width / n

    list1 = p
    list2 = tru
    # 重新设置x轴的坐标
    fig = plt.figure(figsize=(12, 4))
    x = x - (total_width - width) / 2
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.serif']=['Times New Roman']
    # 画柱状图
    fig,ax=plt.subplots()
    ax.bar(x, list1, width=width, label="Pred", color='#0066cc')
    ax.bar(x + width, list2, width=width, label="True", color='#9ACD32')
    # plt.bar(x + 2*width, c, width=width, label="c")
    plt.xticks(np.arange(15), tuple(effect), rotation=90)
    # 显示图例

    plt.legend(loc='lower right')
    plt.title("副作用对比图")
    plt.ylabel("频率")


    st.pyplot(fig)


#---------------------------------------------------------函数定义逻辑----------------------------------------------------

#创建一个文本布局

data_load_state = st.text('正在加载信息...')

#-------------------------------------------------------后端逻辑----------------------------------------------------------
# 定义参数
DF = False
not_FC = False
knn = 5
pca = False
metric = 'cosine'
frequencyMat=load_frequencyMat()
devicee = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
model=load_model(device=devicee)
if pca:
    pca_ = PCA(n_components=256)
    similarity_pca = pca_.fit_transform(frequencyMat)
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

node_label = load_node_label()
feat = torch.tensor(node_label, dtype=torch.float)
sideEffectsGraph = Data(x=feat, edge_index=edges)



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
print(data)
def trans_eng2chi(English):
    translator = LanguageTrans("E2C")
    word = translator.trans(English)
    return word

#supply_effect=[trans_eng2chi(x) for x in data]
supply_effect=data

#-------------------------------------------------------后端逻辑----------------------------------------------------------



#-----------------------------------------------------streamlit逻辑------------------------------------------------------


data_load_state.text('信息加载完毕!')

st.info('副作用预测中，将药物副作用出现频率分为五个等级', icon="ℹ️")
user_choice=st.sidebar.radio('选择查询形式',('根据药物名查询副作用','根据药物分子式查询副作用','查询某药物某副作用大小'))
st.sidebar.warning('1.该软件可根据药物成分及用量预测可能出现的副作用。但是，每个人的身体反应不同，可能会出现意外的反应，请务必在使用药物前咨询医生并阅读药品说明书。')
st.sidebar.warning('2.注意：该软件不适用于预测针对特定症状或疾病的药物。请勿将该软件作为医疗诊断或治疗的替代品。')
st.sidebar.warning('3.本软件不会收集用户的任何个人信息。')
if user_choice=='根据药物名查询副作用':
    search_by_name()
elif user_choice=='根据药物分子式查询副作用':
    search_by_Chemical()
elif user_choice=='查询某药物某副作用大小':
    option = st.selectbox(
        '如何查询药物',
        ('分子式', '药物名'))
    if option=='药物名':
        seach_side_effect_with_name()
    else:
        seach_side_effect_with_chemical()

#--------------------------------------------------------------streamlit逻辑---------------------------------------------
