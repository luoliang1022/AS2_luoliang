from copy import deepcopy
from train_utils import *
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
from utils import *
from train import return_file_name,return_emb1,return_emb2,return_label
from config import Config

config=Config()
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config

        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        #position = PositionalEncoding(d_model, dropout)
        '''
        attncross = MultiHeadedAttention(h, d_model*2)
        ffcross = PositionwiseFeedForward(d_model*2, d_ff, dropout)
        positioncross = PositionalEncoding(d_model*2, dropout)
        '''
        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )
        self.sigmoid=nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = nn.Softmax()

        # Softmax non-linearity

    def forward(self, idx,datasetname):
        embedded_sents1 = return_emb1(idx,datasetname)
        embedded_sents2 = return_emb2(idx,datasetname)
        embedded_sents1 = torch.from_numpy(embedded_sents1).float()
        embedded_sents2 = torch.from_numpy(embedded_sents2).float()


        encoded_sents1 = self.encoder(embedded_sents1, embedded_sents1)
        encoded_sents2 = self.encoder(embedded_sents2, embedded_sents2)



        final_feature_map1 = torch.mean(encoded_sents1, 1)
        final_feature_map2 = torch.mean(encoded_sents2, 1)




        final_out1 = final_feature_map1
        final_out2 = final_feature_map2

        output=self.cos(final_out1 , final_out2)


        return output
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, epoch):
        self.train()

        # Reduce learning rate as number of epochs increase
        #if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
        #   self.reduce_lr()

        for idx in range(0,16000,32):
            self.optimizer.zero_grad()

            y = return_label(idx,"train")
            y = torch.from_numpy(y).float()
            y_pred = self.__call__(idx,"train")

            print("==============")
            print("正在训练第"+str(epoch)+"轮，第"+str(idx/32)+"组训练集")

            loss = self.loss_op(y_pred, y)
            loss.backward()
            self.optimizer.step()

        print("Evaluating Epoch"+str(epoch))
        print("=============================")
        config=Config()
        trainfilename,testfilename,validfilename = return_file_name()

        #train_accuracy = evaluate_model(self, "train",16000, filename = trainfilename)
        val_accuracy,v_c = evaluate_model(self, "valid" ,1148, filename = validfilename)
        #test_accuracy,t_c = evaluate_model(self, "test",1517, filename = testfilename)

        #print("training \t" + str(train_accuracy*config.multiplyby))
        print("validation \t"+str(val_accuracy*config.multiplyby))
        #print("test \t" + str(test_accuracy*config.multiplyby))

        return  val_accuracy, v_c