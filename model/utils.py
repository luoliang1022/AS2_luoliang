import torch
from config import Config
from imap_qa import calc_map1, calc_mrr1
val = 0

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.vocab1 = []
        self.vocab2 = []
        self.word_embeddings = {}
        self.weights = []

    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def read_data(self, filename):
        with open(filename, 'r') as datafile:
            res = []
            count = 0
            for line in datafile:
                count = count + 1
                if (count == 1):
                    continue
                line = line.strip().split('\t')
                lines = []
                length = len(line)
                if (length < 3):
                    lines.append(line[0])
                    lines.append("question")
                    lines.append(line[1])
                else:
                    lines.append(line[0])
                    lines.append(line[1])
                    lines.append(line[2])

                res.append([lines[0], lines[1], float(lines[2])])

        return res

def evaluate_model(model, datasetname,datasetnum, filename):
    all_preds = []
    #all_y = []
    model.eval()
    with torch.no_grad():
        for idx in range(0,datasetnum,32):

            y_pred = model(idx,datasetname)

            predicted = y_pred.cpu().data.numpy()

            all_preds.extend(predicted)

        config = Config()
        datasets = Dataset(config)
        t_f = datasets.read_data(filename=filename)
        score = calc_map1(t_f, all_preds)
        score2 = calc_mrr1(t_f, all_preds)
        print(filename + " MAP: " + str(score * 1))
        print(filename+" MRR: " + str(score2 * 1))
        return score,score2