from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np
import h5py
from contextualized_embedding import Contextualized_Embedding
import torch
import numpy as np
class bert_con_emb:
  def __init__(self, filepath, ce):
    self.filepath = filepath
    self.ce = ce
  def read_data(self,filename):
    with open(filename, 'r',encoding="utf8") as datafile:
      res = []
      count=0
      for line in datafile:
        count=count+1
        if(count==1):
          continue
        #len(line) = 3
        line = line.strip().split('\t')
        res.append([line[0].lower(), line[1].lower(), float(line[2])])
    return res;
  def deal_data(self):
    resx = self.read_data(self.filepath)
    datanum = len(resx)
    #print(len(resx))
    #for i in range(0, datanum):
      #print(resx[i])

    str1 = ""
    str2 = ""
    for i in resx:
      i[0] = i[0].split()
      i[1] = i[1].split()
      separator = ' '
      temp1 = separator.join(i[0])
      temp2 = separator.join(i[1])
      str1 += temp1 + "\n"
      str2 += temp2 + "\n"
    str1 = str1.split('\n')
    str2 = str2.split('\n')
    return datanum,resx,str1,str2
  def write_q(self,datanum,str1,hf1):
    d_model = 768
    #j = 0
    for i in range(0, datanum, 32):
      temp = []
      lenn = 0
      for j in range(0, 32):
        if (i + j >= datanum):
          continue
        result = self.ce.contextualizedEmbedding(str1[i + j])
        temp.append(result)
        lenr = int((result.size) / d_model)
        lenn = max(lenn, lenr)
      print("第" + str(i / 32) + "组长度最长为：" + str(lenn))
      if (lenn > 20):
        lenn = 20
      print("第" + str(i / 32) + "组bitch_size为" + str(len(temp)))
      for j in range(0, len(temp)):
        res = temp[j]
        print("第" + str(i / 32) + "组，第" + str(j) + "个：shape:" + str(res.shape))
        lenr = int((res.size) / d_model)
        if (lenr > lenn):
          xx = res[:, 0:lenn, :]
        else:
          val = lenn - int((res.size) / d_model)
          y = np.zeros((1, val, d_model))
          yy = res
          xx = np.append(yy, y, axis=1)
        print("xx的shape变为：" + str(xx.shape))
        if (j == 0):
          x = xx
        else:
          x = np.append(x, xx, axis=0)
      print("正在写入第" + str(i / 32) + "组问题")
      hf1.create_dataset('dataset_' + str(i), data=x)
      print("第" + str(i / 32) + "组维度为" + str(x.shape))
      print("=======================================================")
    hf1.close()
  def write_a(self,datanum,str2,hf2):
    d_model = 768
    j = 0
    for i in range(0, datanum, 32):
      temp = []
      lenn = 0
      for j in range(0, 32):
        if (i + j >= datanum):
          continue
        result = self.ce.contextualizedEmbedding(str2[i + j])
        temp.append(result)
        lenr = int((result.size) / d_model)
        lenn = max(lenn, lenr)
      print("第" + str(i / 32) + "组长度最长为：" + str(lenn))
      if (lenn > 50):
        lenn = 50
      print("第" + str(i / 32) + "组bitch_size为" + str(len(temp)))
      for j in range(0, len(temp)):
        res = temp[j]
        print("第" + str(i / 32) + "组，第" + str(j) + "个：shape:" + str(res.shape))
        lenr = int((res.size) / d_model)
        if (lenr > lenn):
          xx = res[:, 0:lenn, :]
        else:
          val = lenn - int((res.size) / d_model)
          y = np.zeros((1, val, d_model))
          yy = res
          xx = np.append(yy, y, axis=1)
        print("xx的shape变为：" + str(xx.shape))
        if (j == 0):
          x = xx
        else:
          x = np.append(x, xx, axis=0)
      print("正在写入第" + str(i / 32) + "组答案")
      hf2.create_dataset('dataset_' + str(i), data=x)
      print("第" + str(i / 32) + "组维度为" + str(x.shape))
      print("=======================================================")
    hf2.close()

  def write_label(self,datanum,resx,hf3):
    for i in range(0, datanum, 32):
      list = []
      for j in range(0, 32):
        if (i + j >= datanum):
          continue
        list.append(resx[i + j][2])
      label = np.array(list)
      print("正在写入第" + str(i / 32) + "组标签")
      print("第" + str(i / 32) + "组维度为" + str(label.shape))
      print(label)
      #print(type(label))
      hf3.create_dataset('dataset_' + str(i), data=label)
      '''
      lab = torch.from_numpy(label).float().cuda
      print(lab)
      print(type(lab))
      '''
      print("======================")
    hf3.close()

'''
import sys
from google.colab import drive
drive.mount('/content/drive')
sys.path.append('/content/drive/MyDrive/luoliang/CETEFeature_Based/Model_Transformer/GenerateContextualizedEmbeddings')
'''
#!pip install git+https://github.com/huggingface/transformers



if __name__=='__main__':
  file = "D:/2021/AS2experiment/AS2/model_Transformer/dataSetRaw/TRECR/trecr_train.tsv"
  #valid_file = "/content/drive/MyDrive/luoliang/CETEFeature_Based/Model_Transformer/CETE Dataset/data/trecr_test.tsv"
  #valid_file = "/content/drive/MyDrive/luoliang/CETEFeature_Based/Model_Transformer/CETE Dataset/data/trecr_valid.tsv"


  path = "D:/2021/AS2experiment/AS2/model_Transformer/dataSet/train/"
  #path = "/content/drive/MyDrive/luoliang/CETEFeature_Based/Model_Transformer/emb/trecr_valid/"
  #path = "/content/drive/MyDrive/luoliang/CETEFeature_Based/Model_Transformer/emb/trecr_test/"

  #fileq = "trecrtrain_q.h5"
  #filea = "trecrtrain_aa.h5"
  filelab = "trecrtrain_label.h5"
  '''

  fileq = "trecrvalid_q.h5"
  filea = "trecrvalid_a.h5"
  filelab = "trecrvalid_label.h5"

  '''
  '''
  fileq = "trecrtest_q.h5"
  filea = "trecrtest_a.h5"
  filelab = "trecrtest_label.h5"
  '''
  #hf1 = h5py.File(path+fileq,'w')

  #hf2 = h5py.File(path+filea,'w')

  hf3 = h5py.File(path+filelab,'w')

  t = BertTokenizer.from_pretrained('bert-base-uncased')
  m = BertModel.from_pretrained("bert-base-uncased",output_hidden_states = True,)
  ce = Contextualized_Embedding(m,t)
  b_c_e = bert_con_emb(file,ce)
  datanum,resx,str1,str2 = b_c_e.deal_data()
  print(datanum)
  print(len(str1))
  #b_c_e.write_q(datanum, str1, hf1)
  #b_c_e.write_a(datanum,str2,hf2)
  b_c_e.write_label(datanum,resx,hf3)

