import numpy as np
import pandas as pd
import jieba
import re
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class DataProcessor:

    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def preprocess_data(self):
        text = self.data_cut()
        labels = LabelEncoder().fit_transform(self.labels)
        word2index, sent2indexs = self.text_embedding(text)
        X_train, X_test, y_train, y_test = train_test_split(sent2indexs, labels, test_size=0.2, random_state=666)
        train_dataset = self.make_dataset(X_train, y_train)
        test_dataset = self.make_dataset(X_test, y_test)
        dataloader_train = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
        return word2index, dataloader_train, dataloader_test

    def data_cut(self):
        index = 0
        review_cut = []
        for item in self.data:
            item = re.sub('@.*?:', '', item)
            item = re.sub('@.*?：', '', item)
            item = re.sub(r'\W+', ' ', item).replace('_', ' ')
            cut = jieba.lcut(item)
            segResult = []
            for word in cut:
                if ('\u4e00' <= word <= '\u9fa5'):
                    segResult.append(word)
            review_cut.append(' '.join(segResult))
            index += 1
        return review_cut

    @staticmethod
    def make_dataset(data, labels):
        data = torch.LongTensor(np.array(data))
        labels = torch.LongTensor(np.array(labels))
        return list(zip(data, labels))

    def text_embedding(self, sentences):
        word2index = {"PAD": 0}
        word2index = self.compute_word2index(sentences, word2index)
        sent2indexs = []
        for sent in sentences:
            sentence = self.compute_sent2index(sent, MAX_LEN, word2index)
            sent2indexs.append(sentence)
        return word2index, sent2indexs

    @staticmethod
    def compute_word2index(sentences, word2index):
        for sentence in sentences:
            for word in sentence.split():
                if word not in word2index:
                    word2index[word] = len(word2index)
        return word2index

    @staticmethod
    def compute_sent2index(sentence, max_len, word2index):
        sent2index = [word2index.get(word, 0) for word in sentence.split()]
        if len(sentence.split()) < max_len:
            sent2index += (max_len - len(sentence.split())) * [0]
        else:
            sent2index = sentence[:max_len]
        return sent2index


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, feature_size, windows_size, max_len, n_class):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=h),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=max_len - h + 1),
                          )
            for h in windows_size]
        )
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=feature_size * len(windows_size), out_features=n_class)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = [conv(x) for conv in self.conv1]
        x = torch.cat(x, 1)
        x = x.view(-1, x.size(1))
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, num_classes, device):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)

    def forward(self, x):
        x = self.embed(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.cat([out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]], dim=1)
        x = self.dropout(out)
        x = self.fc(x)
        return x


class Trainer:

    def __init__(self, model, train_loader, test_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs, loss_func):
        for epoch in range(epochs):
            self.model.train()
            train_loss, test_loss = 0, 0
            train_preds, train_targets = [], []
            test_preds, test_targets = [], []
            for i, (input, label) in enumerate(self.train_loader):
                input, label = input.to(self.device), label.to(self.device)
                output = self.model(input)
                loss = loss_func(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred = torch.max(output, 1)[1].cpu().numpy()
                label = label.cpu().numpy()
                train_preds.extend(pred)
                train_targets.extend(label)
                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                for ii, (input, label) in enumerate(self.test_loader):
                    input, label = input.to(self.device), label.to(self.device)
                    output = self.model(input)
                    loss = loss_func(output, label)

                    pred = torch.max(output, 1)[1].cpu().numpy()
                    label = label.cpu().numpy()
                    test_preds.extend(pred)
                    test_targets.extend(label)
                    test_loss += loss.item()

                train_acc = accuracy_score(train_targets, train_preds)
                test_acc = accuracy_score(test_targets, test_preds)
                train_precision = precision_score(train_targets, train_preds, average='macro')
                train_recall = recall_score(train_targets, train_preds, average='macro')
                train_f1 = f1_score(train_targets, train_preds, average='macro')

                test_precision = precision_score(test_targets, test_preds, average='macro')
                test_recall = recall_score(test_targets, test_preds, average='macro')
                test_f1 = f1_score(test_targets, test_preds, average='macro')

                print(f"Epoch: {epoch + 1}, train_loss: {train_loss / (i + 1):.5f}, test_loss: {test_loss / (ii + 1):.5f}, train_acc: {train_acc * 100:.2f}%, test_acc: {test_acc * 100:.2f}%")
                print(f"Train precision: {train_precision:.4f}, Train recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")
                print(f"Test precision: {test_precision:.4f}, Test recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")
                print("-------------------------------------------------------------------")

if __name__ == '__main__':
    # 参数设置
    BATCH_SIZE = 32 # 批处理大小
    EPOCHS = 10 # 训练轮数
    WINDOWS_SIZE = [2, 4, 3]    # 卷积核大小
    MAX_LEN = 100   # 文本最大长度
    EMBEDDING_DIM = 600 # 词向量维度
    FEATURE_SIZE = 200  # 卷积核数量
    N_CLASS = 2 # 类别数量
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 设备
    loss_func = nn.CrossEntropyLoss()   # 损失函数

    # 数据准备
    dataset = pd.read_csv('./data/weibo_senti_10k.csv', delimiter=',')
    processor = DataProcessor(dataset['review'].tolist(), dataset['label'].tolist(), BATCH_SIZE)
    word2index, dataloader_train, dataloader_test = processor.preprocess_data()

    # 构建模型

    model = TextCNN(vocab_size=len(word2index), embedding_dim=EMBEDDING_DIM, windows_size=WINDOWS_SIZE,
                    max_len=MAX_LEN, feature_size=FEATURE_SIZE, n_class=N_CLASS).to(DEVICE)
    # model = BiLSTM(num_embeddings=len(word2index), embedding_dim=EMBEDDING_DIM, hidden_size=FEATURE_SIZE,
    #                  num_layers=2, num_classes=N_CLASS, device=DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, dataloader_train, dataloader_test, optimizer, DEVICE)

    # 模型训练
    trainer.train(EPOCHS, nn.CrossEntropyLoss())

    # 模型保存
    torch.save(model.state_dict(), './data/text.pkl')
