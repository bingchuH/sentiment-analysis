from collections import defaultdict
import jieba
import re
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SentimentAnalyzer:
    def __init__(self, dict_choice='Boson'):
        self.dict_choice = dict_choice
        self.degree_dict = self.load_dict('degree')
        self.inverse_words = self.load_dict('inverse')
        self.stopwords = self.load_dict('stopwords')
        if dict_choice == 'Boson':
            self.sentiment_dict = self.load_dict('sentiment')
        elif dict_choice == 'NTUSD':
            self.positive_words = self.load_dict('positive')
            self.negative_words = self.load_dict('negative')

    def load_dict(self, request):
        path = './dictionary/'
        if request == 'degree':
            degree_file = open(path + 'degree.txt', 'r+', encoding='utf-8')
            degree_list = degree_file.readlines()
            degree_dict = defaultdict()
            for i in degree_list:
                degree_dict[i.split(',')[0]] = float(i.split(',')[1])
            return degree_dict
        elif request == 'sentiment':
            with open(path + 'sentiment_score.txt', 'r+', encoding='utf-8') as f:
                sen_list = f.readlines()
                sen_dict = defaultdict()
                for i in sen_list:
                    if len(i.split(' ')) == 2:
                        sen_dict[i.split(' ')[0]] = float(i.split(' ')[1])
            return sen_dict
        elif request == "positive":
            file = open(path + "positive_simplified.txt", encoding='utf-8')
        elif request == "negative":
            file = open(path + "negative_simplified.txt", encoding='utf-8')
        elif request == "inverse":
            file = open(path + "inverse_words.txt", encoding='utf-8')
        elif request == 'stopwords':
            file = open(path + 'stopwords.txt', encoding='utf-8')
        else:
            return None
        dict_list = []
        for word in file:
            dict_list.append(word.strip())
        return dict_list

    def data_cut(self, review):
        index = 0
        review_cut = []
        for item in review:
            # 删除用户名
            item = re.sub('@.*?:', '', item)
            item = re.sub('@.*?：', '', item)
            # 删除特殊字符
            item = re.sub(r'\W+', ' ', item).replace('_', ' ')
            # 分词
            cut = jieba.lcut(item)
            segResult = []
            # 判断非中文字符串如链接等
            for word in cut:
                if ('\u4e00' <= word <= '\u9fa5'):
                    segResult.append(word)
            review_cut.append(' '.join(segResult))
            index += 1
        return review_cut


    def classify_words_pn(self, word_list):
        z = 0  # 记录程度副词位置
        score = []  # 记录情感词分数
        for word_index, word in enumerate(word_list):
            w = 0  # 当前词的情感分数
            if word in self.positive_words:  # 为正面情感词
                w += 1
                for i in range(z, int(word_index)):  # 遍历当前词之前的词是否有否定词、程度词
                    if word_list[i] in self.inverse_words:  # 如果有否定词
                        w = w * (-1)
                        for j in range(z, i):  # 程度词+否定词+情感词
                            if word_list[j] in self.degree_dict:
                                w = w * 2 * self.degree_dict[word_list[j]]
                                break
                        for j in range(i, int(word_index)):  # 否定词+程度词+情感词
                            if word_list[j] in self.degree_dict:
                                w = w * 0.5 * self.degree_dict[word_list[j]]
                                break
                    elif word_list[i] in self.degree_dict:
                        w = w * self.degree_dict[word_list[i]]
                z = int(word_index) + 1
            if word in self.negative_words:  # 为负面情感词
                w -= 1
                for i in range(z, int(word_index)):
                    if word_list[i] in self.inverse_words:
                        w = w * (-1)
                        for j in range(z, i):  # 程度词+否定词+情感词
                            if word_list[j] in self.degree_dict:
                                w = w * 2 * self.degree_dict[word_list[j]]
                                break
                        for j in range(i, int(word_index)):  # 否定词+程度词+情感词
                            if word_list[j] in self.degree_dict:
                                w = w * 0.5 * self.degree_dict[word_list[j]]
                                break
                    elif word_list[i] in self.degree_dict:
                        w *= self.degree_dict[word_list[i]]
                z = int(word_index) + 1
            score.append(w)
        score = sum(score)
        return score

    def classify_words_value(self, word_list):
        scores = []
        z = 0
        for word_index, word in enumerate(word_list):
            score = 0
            if word in self.sentiment_dict.keys() and word not in self.inverse_words and word not in self.degree_dict.keys():
                score = self.sentiment_dict[word]
                for i in range(z, int(word_index)):  # 遍历当前词之前的词是否有否定词、程度词
                    if word_list[i] in self.inverse_words:  # 如果有否定词
                        score = score * (-1)
                        for j in range(z, i):  # 程度词+否定词+情感词
                            if word_list[j] in self.degree_dict:
                                score = score * self.degree_dict[word_list[j]] * 2
                                break
                        for j in range(i, int(word_index)):  # 否定词+程度词+情感词
                            if word_list[j] in self.degree_dict:
                                score = score * self.degree_dict[word_list[j]] * 0.5
                                break
                    elif word_list[i] in self.degree_dict:
                        score = score * float(self.degree_dict[word_list[i]])
                z = int(word_index) + 1
            scores.append(score)
        scores = sum(scores)
        return scores


    def calculate_scores_and_predicts(self, review_cut):
        scores, predicts = [], []
        for i, sentence in enumerate(review_cut):
            word_list = [x.strip() for x in sentence.split(' ')]
            if self.dict_choice == 'Boson':
                score = self.classify_words_value(word_list)
            elif self.dict_choice == 'NTUSD':
                score = self.classify_words_pn(word_list)
            scores.append(score)
            predicts.append(1 if score > 0 else 0)
        return scores, predicts

    def evaluate(self, label, predicts):
        accuracy = accuracy_score(label, predicts)
        precision = precision_score(label, predicts)
        recall = recall_score(label, predicts)
        f1 = f1_score(label, predicts)
        print('准确率：', accuracy, '\n正确率：', precision, '\n召回率：', recall, '\nF1值：', f1)


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('./data/weibo_senti_10k.csv')
    review, label = data['review'].values, data['label'].values

    analyzer = SentimentAnalyzer()
    review_cut = analyzer.data_cut(review)
    scores, predicts = analyzer.calculate_scores_and_predicts(review_cut)
    analyzer.evaluate(label, predicts)
