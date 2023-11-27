import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import string  # Add this import statement

class TextClassifier:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, encoding='latin-1')
        self.ham_words = ''
        self.spam_words = ''
        self.pred_scores_word_vectors = []

    def create_wordcloud(self, text, color):
        wordcloud = WordCloud(width=500, height=300).generate(text)
        plt.figure(figsize=(10, 8), facecolor=color)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    def preprocess_data(self):
        self.data = self.data.replace(['ham', 'spam'], [0, 1])

        nltk.download('stopwords')

        def text_process(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
            return " ".join(text)

        self.data['text'] = self.data['text'].apply(text_process)

    def generate_wordclouds(self):
        # Ensure data has been preprocessed
        if 'text' not in self.data.columns:
            print("Data has not been preprocessed. Please call preprocess_data() first.")
            return

        for label in ['spam', 'ham']:
            messages = self.data[self.data['label'] == label].text
            if not messages.empty:  # Check if there are messages for the label
                text = ' '.join([val.lower() for val in messages])
                color = 'w' if label == 'spam' else 'g'
                self.create_wordcloud(text, color)
            else:
                print(f"No messages found for label '{label}'. Skipping word cloud generation.")

    def build_word_vectors(self):
        total_counts = Counter()
        for text_ in self.data['text']:
            for word in text_.split(" "):
                total_counts[word] += 1

        vocab = sorted(total_counts, key=total_counts.get, reverse=True)
        vocab_size = len(vocab)
        word2idx = {}

        def text_to_vector(text):
            word_vector = np.zeros(vocab_size)
            for word in text.split(" "):
                if word2idx.get(word) is None:
                    continue
                else:
                    word_vector[word2idx.get(word)] += 1
            return np.array(word_vector)

        word_vectors = np.zeros((len(self.data['text']), len(vocab)), dtype=np.int_)
        for i, text_ in enumerate(self.data['text']):
            word_vectors[i] = text_to_vector(text_)

        return word_vectors

    def build_tfidf_vectors(self):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(self.data['text'])
        return vectors

    def train_predict_models(self, features, targets):
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.15, random_state=111)

        svc = SVC(kernel='sigmoid', gamma=1.0)
        knc = KNeighborsClassifier(n_neighbors=49)
        mnb = MultinomialNB(alpha=0.2)
        dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
        lrc = LogisticRegression(solver='liblinear', penalty='l1')
        rfc = RandomForestClassifier(n_estimators=31, random_state=111)

        clfs = {'SVC': svc, 'KN': knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}

        for k, v in clfs.items():
            v.fit(X_train, y_train)
            pred = v.predict(X_test)
            self.pred_scores_word_vectors.append((k, [accuracy_score(y_test, pred)]))

data_path = 'C:\\Users\\yjun0\\OneDrive - Asia Pacific University\\A.P.U\\Projects\\spam_ham_dataset.csv'
text_classifier = TextClassifier(data_path)
text_classifier.preprocess_data()
word_vectors = text_classifier.build_word_vectors()
