
import h5py
from model import *
import numpy as np
from config import Config
import torch.optim as optim
from torch import nn
import torch

def return_file_name():
    config = Config()
    train_file = config.dataraw + config.data + '_train.tsv'
    test_file = config.dataraw + config.data + '_test.tsv'
    valid_file = config.dataraw + config.data + '_valid.tsv'
    return train_file,test_file,valid_file

def return_emb1(k,setname):
    config = Config()
    outfile = "";
    if setname=="train":
        outfile = config.trainpath + config.data +setname+ "_q"
    elif setname=="valid":
        outfile = config.validpath + config.data+setname + "_q"
    elif setname=="test":
        outfile = config.testpath + config.data+setname + "_q"
    hf = h5py.File(outfile + ".h5", 'r')
    n1 = hf.get('dataset_' + str(k))
    n1 = np.array(n1)
    return n1

def return_emb2(k,setname):
    config = Config()
    outfile = ""
    if setname=="train":
        outfile = config.trainpath + config.data+setname + "_a"
    elif setname=="valid":
        outfile = config.validpath + config.data+setname + "_a"
    elif setname=="test":
        outfile = config.testpath + config.data+setname + "_a"
    hf = h5py.File(outfile + ".h5", 'r')
    n1 = hf.get('dataset_' + str(k))
    n1 = np.array(n1)
    return n1
def return_label(k,setname):
    config = Config()
    outfile = ""
    if setname=="train":
        outfile = config.trainpath + config.data + setname + "_label"
    elif setname=="valid":
        outfile = config.validpath + config.data + setname + "_label"
    elif setname=="test":
        outfile = config.testpath + config.data + setname + "_label"
    hf = h5py.File(outfile + ".h5", 'r')
    n1 = hf.get('dataset_' + str(k))
    n1 = np.array(n1)
    return n1;

if __name__=='__main__':
    config = Config()

    if(config.model==0):
        model = Transformer(config)

    if torch.cuda.is_available():
        model.cuda()
        #https://blog.csdn.net/Qy1997/article/details/106455717
    model.train()




    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    MSELoss = nn.MSELoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(MSELoss)
    print("##############################################################")

    train_losses = []
    val_accuracies = []
    val_map=0
    max_val=0

    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        val_map, val_mrr = model.run_epoch(i)  # i
        if(val_map>=max_val):
            max_val=val_map
            best_model = deepcopy(model)
        val_accuracies.append(val_map)


    #---------------------------------------------------------------------
    print("现在开始测试集验证")
    print("===========================")
    train_file, test_file, valid_file = return_file_name()
    test_acc,t_acc = evaluate_model(best_model, "test",1517, filename = test_file)



    #-------------------------------
    print('#Final Test MAP' + "\t" + str(test_acc*config.multiplyby))
    print('#Final Test MRR' + "\t" + str(t_acc * config.multiplyby))
