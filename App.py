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
path=r"Supplementary Data 1.txt"
pic_path=r"17593290.png"
model_path=r'net_params.pth'
frequence_path=r'frequencyMat.csv'
side_effect_path=r'side_effect_label_750.mat'
raw_frequency_path=r'raw_frequency_750.mat'

frequence=['none','Very rare','Rare',' Infrequent','Frequent','Very frequent']
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
    #translator = LanguageTrans("C2E")

    #word = translator.trans(chinese)
    word = chinese
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
st.set_page_config(page_title = 'Prediction of the frequency of medication side effects',page_icon = '🕵️‍♀️',layout = 'wide',initial_sidebar_state='expanded')

image_path = "17593290.png"
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


st.title('Prediction of the frequency of medication side effects')

st.sidebar.expander('')
st.sidebar.write('User input ⬇️')
Top_K=st.sidebar.slider(label='Select the top K side effects',min_value=1,max_value=25,value=(1,5))[1]

#---------------------------------------------------创建标题-------------------------------------------------------------


#-------------------------------------------------函数定义逻辑------------------------------------------------------------
#@st.cache
def load_model():
    side_effect_label = r"side_effect_label_750.mat"
    input_dim = 109
    model = GAT3().to(device=device)
    model.load_state_dict(torch.load(model_path,map_location='cpu'), strict=False)
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
    medecine_name = st.text_input('Please enter the name of the medication in the text box below：', value='Please click here to enter')

    try:
        Eng_name = translate_name(medecine_name)
    except Exception as e:
        Eng_name = None

    if Eng_name == None:
        Smiles = None
    else:
        Smiles = get_Smiles(Eng_name)

    # 构造API请求,点击确定获取药物分子式并进行预测
    if st.button('Start prediction'):

        if Smiles != None:
            state = st.text('Start medication analysis...')
            st.write('The SMILES molecular formula of the medication is:', Smiles)
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(50):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'The progress of the analysis:{i * 2 + 2}%')
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

            dataframe=pd.DataFrame(temp_result,index=data,columns=['Frequency rating'])





            # ----------------------------------------预测药物代码-----------------------

            # 显示进度

            # dataframe排序
            dataframe.sort_values(by='Frequency rating', inplace=True, ascending=False)

            result=list(dataframe['Frequency rating'])

            class_=[frequence[score] for score in result]

            dataframe2=pd.DataFrame(class_,index=dataframe.index,columns=['Frequency rating'])
            # 加载完成
            st.success('Medication analysis is completed!', icon="✅")

            col1, col2 = st.columns(2)

            with col1:
                draw_topK(Top_K, dataframe)

          
                st.info('The molecular data of the medication is sourced from an open-access website:https://pubchem.ncbi.nlm.nih.gov/。',
                        icon="ℹ️")
            with col2:
                pic = draw_chaimcal(Smiles)

                #st.header('Displaying the molecular diagram of the medication')

                #st.image(pic, caption=Smiles)

                st.header('Displaying the medication scoring table')
                st.dataframe(dataframe2.style.highlight_max(axis=0))
            #draw_comparation(prob.cpu().detach())


        else:

            st.warning('No medication found. This could be due to the input medication not being included in the database or a problem with your network. Please check your network connection or try again with a different input.', icon="⚠️")

def search_by_Chemical():
    chamical_name = st.text_input('Please enter the molecular formula of the medication here: ', value='Please click here to enter')
    if st.button('Start prediction'):
        if chamical_name != None:
            state = st.text('Analyzing the medication...')
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(50):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'The progress of the analysis:{i * 2 + 2}%')
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

            dataframe = pd.DataFrame(temp_result, index=data, columns=['Frequency rating'])

            # ----------------------------------------预测药物代码-----------------------
            # 显示进度

            # dataframe排序
            dataframe.sort_values(by='Frequency rating', inplace=True, ascending=False)

            result=list(dataframe['Frequency rating'])

            class_=[frequence[score] for score in result]

            dataframe2=pd.DataFrame(class_,index=dataframe.index,columns=['Frequency rating'])


            # 加载完成
            st.success('Medication analysis is completed!', icon="✅")
            col1, col2 = st.columns(2)
            with col1:
                draw_topK(Top_K, dataframe)



                
                st.info('The molecular data of the medication is sourced from an open-access website:https://pubchem.ncbi.nlm.nih.gov/',
                        icon="ℹ️")

            with col2:
                #pic = draw_chaimcal(chamical_name)

                #st.header('Displaying the molecular diagram of the medication')

                #st.image(pic, caption=chamical_name)
                st.header('Displaying the medication scoring table')
                st.dataframe(dataframe2.style.highlight_max(axis=0))

        else:

            st.warning('No medication found. This could be due to the input medication not being included in the database or a problem with your network. Please check your network connection or try again with a different input.',icon="⚠️")

def seach_side_effect_with_name():
    # 输入药物名
    medecine_name = st.text_input('Please enter the molecular formula of the medication here: ', value='Please click here to enter')
    side_effect_name=st.text_input('Please enter the name of the side effect here:',value='Please click here to enter')

    try:
        Eng_name = translate_name(medecine_name)
    except Exception as e:
        Eng_name = None

    if Eng_name == None:
        Smiles = None
    else:
        Smiles = get_Smiles(Eng_name)
    if st.button('Confirmed'):
        if Smiles != None:
            state = st.text('Start medication analysis...')
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(50):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Analysis progress:{i * 2 + 2}%')
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
            dataframe = pd.DataFrame(result, index=data, columns=['Frequency rating'])
            # ----------------------------------------预测药物代码-----------------------

            # 显示进度

            # dataframe排序
            dataframe.sort_values(by='Frequency rating', inplace=True, ascending=False)

            # 加载完成
            st.success('Medication analysis is completed! ', icon="✅")


            st.write('Side effect frequency rating is：',dataframe.loc[side_effect_name,'Frequency rating'])

            
            st.info('The molecular data of the medication is sourced from an open-access website:https://pubchem.ncbi.nlm.nih.gov/', icon="ℹ️")

        else:

            st.warning('No medication found. This could be due to the input medication not being included in the database or a problem with your network. Please check your network connection or try again with a different input.', icon="⚠️")

def seach_side_effect_with_chemical():
    # 输入药物名
    medecine_name = st.text_input('Please enter the molecular formula of the medication here: ', value='Please click here to enter')
    side_effect_name = st.text_input('Please enter the name of the side effect here:',value='Please click here to enter')
    if st.button('Confirmed'):
        Smiles = get_Smiles(medecine_name)

        if Smiles != None:
            state = st.text('Start medication analysis...')
            st.write('The SMILES molecular formula of the medication is:', Smiles)
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(50):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Progress of analysis:{i * 2 + 2}%')
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
            dataframe = pd.DataFrame(result, index=data, columns=['Frequency rating'])
            #dataframe = pd.DataFrame(temp_result, index=data, columns=['Frequency rating'])
            # ----------------------------------------预测药物代码-----------------------

            # 显示进度

            # dataframe排序
            dataframe.sort_values(by='Frequency rating', inplace=True, ascending=False)

            # 加载完成
            st.success('Medication analysis is completed!', icon="✅")

            st.write('Side effect frequency rating is：', dataframe.loc[side_effect_name, 'Frequency rating'])

        else:

            st.warning('No medication found. This could be due to the input medication not being included in the database or a problem with your network. Please check your network connection or try again with a different input.', icon="⚠️")

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
    st.header('Bar Chart of Side Effect Ratings')
    fig = go.Figure(
        data=[go.Bar(x=sub_df.index, y=sub_df['Frequency rating'], name='Side Effect Rating')],
        layout=go.Layout(
            title='',
            xaxis=dict(title='Sample ID'),
            yaxis=dict(title='Rating'),
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
    plt.title("Side Effects Comparison Chart")
    plt.ylabel("Frequency rating")


    st.pyplot(fig)


#---------------------------------------------------------函数定义逻辑----------------------------------------------------

#创建一个文本布局

data_load_state = st.text('Loading information...')

#-------------------------------------------------------后端逻辑----------------------------------------------------------
# 定义参数
DF = False
not_FC = False
knn = 5
pca = False
metric = 'cosine'
frequencyMat=load_frequencyMat()
devicee = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
model=load_model()
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


data_load_state.text('Information loading complete! ')

st.info('During the side effect prediction, the frequency of occurrence of medication side effects is divided into five levels', icon="ℹ️")
user_choice=st.sidebar.radio('Select the query format',('Search for side effects based on the medication name','Search for side effects based on the medication molecular formula','Inquire about the magnitude of a specific side effect for a particular medication'))
st.sidebar.warning('1.This software can predict possible side effects based on the medication\'s ingredients and dosage. However, everyone\'s body reacts differently, and unexpected reactions may occur. It is important to consult a doctor and read the medication\'s instructions before using it.')
st.sidebar.warning('2.Note: This software is not suitable for predicting medications specifically targeting certain symptoms or diseases. Please do not consider this software as a substitute for medical diagnosis or treatment.')
st.sidebar.warning('3.This software does not collect any personal information from users.')
if user_choice=='Search for side effects based on the medication name':
    search_by_name()
elif user_choice=='Search for side effects based on the medication molecular formula':
    search_by_Chemical()
elif user_choice=='Inquire about the magnitude of a specific side effect for a particular medication':
    option = st.selectbox(
        'How to search for medications',
        ('Molecular formula', 'Medication name'))
    if option=='Medication name':
        seach_side_effect_with_name()
    else:
        seach_side_effect_with_chemical()

#--------------------------------------------------------------streamlit逻辑---------------------------------------------
