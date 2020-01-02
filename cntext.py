import os
import json


def format_text():
    myfile = open("cntext.json", 'w', encoding='utf-8')
    dir = '../fudanDatabase'
    for _, dirs, files in os.walk(dir):
        for a in dirs:
            for dir_name, _, sub_files in os.walk(dir + '/' + a):
                print(a)
                i = 0
                for sub_file in sub_files:
                    # print(sub_file)
                    with open(dir_name + '/' + sub_file, 'r', encoding='gb2312', errors='ignore') as f:
                        str = f.read()
                        category = a
                        dic = {'context': str, 'category': a}
                        json_data = json.dumps(dic, ensure_ascii=False)
                        myfile.write(json_data)
                        myfile.write('\n')
                    i += 1
                print(i)

import jieba


def generate_wordlist():
    worddict = {}
    lendict ={}
    cntext = 'cntext.json'
    stopword = 'stopword.txt'
    labelword = ['C31-Enviornment', 'C32-Agriculture', 'C34-Economy', 'C38-Politics', 'C39-Sports']
    stopwords = open(stopword, 'r', encoding='utf-8').read().split('\n')
    wordLabelFile = 'wordLabel.txt'
    lengthFile = 'length.txt'
    file = open(cntext, 'r', encoding='utf-8')
    dataline = file.readline()
    i = 0
    while dataline:
        line = json.loads(dataline)
        if line['category'] in labelword:
            i += 1
            context_seg = jieba.cut(line['context'], cut_all=False)
            print(i)
            length = 0
            for word in context_seg:
                if word in stopwords:
                    continue
                length += 1
                if word in worddict:
                    worddict[word] += 1
                else:
                    worddict[word] = 1
            if length  in lendict:
                lendict[length] += 1
            else:
                lendict[length] = 1
        dataline = file.readline()
    data_num = i
    wordlist = sorted(worddict.items(), key=lambda item:item[1], reverse=True)
    with open(wordLabelFile, 'w', encoding='utf-8') as f:
        ind = 1
        for word in wordlist:
            line = word[0] + " " + str(ind) + ' ' + str(word[1]) + '\n'
            ind += 1
            f.write(line)
    with open(lengthFile, 'w') as f:
        for k, v in lendict.items():
            lendict[k] = round(v * 1.0 / data_num, 3)
        len_list = sorted(lendict.items(), key=lambda item: item[0], reverse=True)
        for t in len_list:
            d = str(t[0]) + ' ' + str(t[1]) + '\n'
            f.write(d)

def read_label(file):
    w2n = {}
    n2w = {}
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            strs = line.split(" ")
            w = strs[0]
            n = strs[1]
            w2n[w] = n
            n2w[n] = w
            line = f.readline()
    return w2n, n2w


def str2vec(str, stopwords, w2n):
    vec = []
    seg = jieba.cut(str, cut_all=False)
    for w in seg:
        if w in stopwords or w not in w2n.keys():
            continue
        else:
            vec.append(w2n[w])
    return vec


def content2vec():
    cntext = 'cntext.json'
    cnvec = 'cnvec.txt'
    labelword = ['C31-Enviornment', 'C32-Agriculture', 'C34-Economy', 'C38-Politics', 'C39-Sports']
    stopword = 'stopword.txt'
    labelnumdic = {}
    labelnum = 800
    stopwords = open(stopword, 'r', encoding='utf-8').read().split('\n')
    stopwords.append('\n')
    wordLabelFile = 'wordLabel.txt'
    lengthFile = 'length.txt'
    w2n, n2w = read_label(wordLabelFile)
    train_data_row = open(cntext, 'r', encoding='utf-8')
    train_data_vec = open(cnvec, 'w', encoding='utf-8')
    line = train_data_row.readline()
    while line:
        json_line = json.loads(line)
        if json_line['category'] in labelword:
            if json_line['category'] in labelnumdic:
                if labelnumdic[json_line['category']] < labelnum:
                    labelnumdic[json_line['category']] += 1
                    vec = str2vec(json_line['context'], stopwords, w2n)
            else:
                labelnumdic[json_line['category']] = 1
                vec = str2vec(json_line['context'], stopwords, w2n)
            vec.append(labelword.index(json_line['category']))
            for num in vec:
                train_data_vec.write(str(num) + ',')
            train_data_vec.write('\n')
        line = train_data_row.readline()
    train_data_vec.close()
    train_data_row.close()


def sentence_to_vector(sentences, w2n, sen_len):

    stopword = '../cndata/stopword.txt'
    stopwords = open(stopword, 'r', encoding='utf-8').read().split('\n')
    stopwords.append('\n')
    vecs = []
    for sentence in sentences:
        vec_pad = [0] * sen_len
        vec = str2vec(sentence, stopwords, w2n)
        if len(vec) > sen_len:
            vec_pad = vec[:sen_len]
        else:
            vec_pad[:len(vec)] = vec
        vecs.append(vec_pad)
    return vecs


from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"
class cn_data(Dataset):
    def __init__(self, data, sen_len):
        self.data = data
        self.sen_len = sen_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = list(filter(is_n, line.split(',')))
        line = [int(x) for x in line]
        label = line[-1]
        data = np.zeros(self.sen_len)
        length = len(line)
        if length - 1 < self.sen_len:
            data[:length - 1] = line[:-1]
        else:
            data[:] = line[:self.sen_len]
        return data, label


def is_n(x):
    return x != '\n'



class LstmImdb(nn.Module):
    def __init__(self, vocab, hidden_size, num_layer, classes):
        super(LstmImdb, self).__init__()
        self.embed = nn.Embedding(vocab, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layer)
        self.fc = nn.Linear(hidden_size, classes)
        self.soft = nn.LogSoftmax(dim=-1)
        self.nl = num_layer
        self.hidden_size = hidden_size

    def forward(self, x):
        embedding = self.embed(x)
        batch_size = x.size()[1]
        h0 = c0 = Variable(embedding.data.new(*(self.nl, batch_size, self.hidden_size)).zero_())
        out, _ = self.lstm(embedding, (h0, c0))
        out = out[:, -1]
        out = self.fc(out)
        # out = F.dropout(out)
        # out = self.soft(out)
        return out


class ConvOned(nn.Module):
    def __init__(self, vocab, hidden_size, classes, kernal_size, max_length):
        super(ConvOned, self).__init__()
        self.embed = nn.Sequential(
            nn.Embedding(vocab, hidden_size),
            nn.Conv1d(max_length, hidden_size, kernal_size),
            nn.AdaptiveAvgPool1d(10),
        )
        self.fc = nn.Sequential(
            nn.Linear(10*hidden_size, classes),
            # nn.Dropout(),
            # nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x.view(x.size()[0], -1))
        return x


def train(model, dataloader, opt, criterion, is_cuda, datalen):
    run_loss = 0
    correct = 0
    label_all = torch.zeros(datalen)
    pre_all = torch.zeros(datalen)
    start = 0

    for i, data in enumerate(dataloader):
        input, label = data
        if is_cuda:
            input, label = input.long().cuda(), label.long().cuda()
        model.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        opt.step()
        run_loss += loss.item()
        pre = torch.max(output, dim=-1)[1]
        label_all[start:start+len(label)] = label
        pre_all[start:start+len(label)] = pre
        start = start + len(label)
        correct += torch.eq(pre, label).sum().item()
    print('train loss:%f, acc:%f %d/%d' % (run_loss/len(dataloader), correct/datalen, correct, datalen))
    print(confusion_matrix(label_all, pre_all))


def test(model, dataloader, criterion, is_cuda, datalen):
    run_loss = 0
    correct = 0
    label_all = torch.zeros(datalen)
    pre_all = torch.zeros(datalen)
    start = 0

    for i, data in enumerate(dataloader):
        input, label = data
        if is_cuda:
            input, label = input.long().cuda(), label.long().cuda()
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, label)
            run_loss += loss.item()
            pre = torch.max(output, dim=-1)[1]
            label_all[start:start + len(label)] = label
            pre_all[start:start + len(label)] = pre
            start = start + len(label)
            correct += torch.eq(pre, label).sum().item()
    print('test loss:%f, acc:%f %d/%d' % (run_loss/len(dataloader), correct/datalen, correct, datalen))
    print(confusion_matrix(label_all, pre_all))

def main():
    dataname = '../cndata/cnvec.txt'
    labelname = '../cndata/wordLabel.txt'
    f = open(dataname, 'r', encoding='utf-8').readlines()
    random.shuffle(f)
    data_len = len(f)
    trainlist = f[:int(data_len*0.8)]
    vallist = f[int(data_len*0.8):]
    sen_len = 200
    traindata = cn_data(trainlist, sen_len)
    valdata = cn_data(vallist, sen_len)
    trainload = DataLoader(traindata, batch_size=256, shuffle=True)
    valload = DataLoader(valdata, batch_size=256, shuffle=True)
    vocab = len(open(labelname, 'r', encoding='utf-8').readlines()) + 1
    # model = LstmImdb(vocab, 200, 2, 5)
    model = ConvOned(vocab, 200, 5, 3, sen_len)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    epoch = 10
    for i in range(epoch):
        print('epoch:%d' % i)
        train(model, trainload, opt, criterion, is_cuda, len(traindata.data))
        test(model, valload, criterion, is_cuda, len(valdata.data))
    print(len(f))
    torch.save(model.state_dict(), "../save/cntext_%d" % epoch)

def predict(model, vecs):
    input = np.array(vecs, dtype=np.long)
    input = torch.from_numpy(input).long()
    print(input.shape)
    with torch.no_grad():
        pre = model(input)
        label_index = torch.max(pre, 1)[1]
    return pre, label_index

def dev():
    from collections import OrderedDict
    labelword = ['C31-Enviornment', 'C32-Agriculture', 'C34-Economy', 'C38-Politics', 'C39-Sports']
    labelname = '../cndata/wordLabel.txt'
    vocab = len(open(labelname, 'r', encoding='utf-8').readlines()) + 1
    sen_len = 200
    model = ConvOned(vocab, 200, 5, 3, sen_len)
    state_dict = torch.load("../save/cntext_10")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    sentences = []
    dirname = "../fudanDatabase/C34-Economy"
    files = os.listdir(dirname)
    for file in files:
        with open(dirname + '/' + file, mode='r', encoding='gb2312', errors='ignore') as f:
            sentences.append(f.read())
    print(len(sentences))
    print(sentences[0])
    wordLabelFile = '../cndata/wordLabel.txt'
    w2n, n2w = read_label(wordLabelFile)
    vecs = sentence_to_vector(sentences, w2n, sen_len)
    pre, label_index = predict(model, vecs)
    result = []
    for idx in label_index.tolist():
        result.append(labelword[idx])
    print(result)



if __name__ == '__main__':
    dev()
