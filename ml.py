import re
import pandas as pd
import jieba
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB


class SentimentAnalysis:

    def __init__(self, file_path, stopword_file):
        self.file_path = file_path
        self.stopword_file = stopword_file
        self.stopwords = self.load_stopwords()

    def load_stopwords(self):
        stopwords = []
        with open(self.stopword_file, encoding='utf-8') as f:
            for word in f:
                stopwords.append(word.strip())
        return stopwords

    def preprocess_data(self):
        # 读取文件
        with open(self.file_path, 'r', encoding='UTF-8') as f:
            data = pd.read_csv(f)
            labels = data['label'].tolist()
            review = data['review'].tolist()

        review_cut = self.data_cut(review)
        data['review_cut'] = review_cut
        data['del_stopwords'] = data['review_cut'].apply(self.del_stopwords)
        data = data.dropna()
        review_cut_list = data['review_cut'].tolist()
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform((d for d in review_cut_list))
        return train_test_split(X, labels, test_size=0.2, random_state=666)

    def data_cut(self, review):
        review_cut = []
        for item in review:
            item = re.sub('@.*?:', '', item)
            item = re.sub('@.*?：', '', item)
            item = re.sub(r'\W+', ' ', item).replace('_', ' ')
            cut = jieba.lcut(item)
            segResult = [word for word in cut if ('\u4e00' <= word <= '\u9fa5')]
            review_cut.append(' '.join(segResult))
        return review_cut

    def del_stopwords(self, words):
        w = [word for word in words.split(' ') if word not in self.stopwords]
        return ' '.join(w)

    @staticmethod
    def train_and_evaluate(classifier, X_train, y_train, X_test, y_test):
        model = classifier()
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')

        print(f"Model: {classifier.__name__}")
        print(f"Train score: {train_score:.4f}")
        print(f"Test score: {test_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("---------------------------")


if __name__ == "__main__":
    file_path = './data/weibo_senti_10k.csv'
    stopword_file = './dictionary/stopwords.txt'
    sa = SentimentAnalysis(file_path, stopword_file)
    X_train, X_test, y_train, y_test = sa.preprocess_data()

    classifiers = [LinearSVC, MultinomialNB, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier]
    for classifier in classifiers:
        SentimentAnalysis.train_and_evaluate(classifier, X_train, y_train, X_test, y_test)
