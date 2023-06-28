from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from html import unescape
import re
import numpy as np
from sklearn.model_selection import train_test_split
import email.policy
import email
import os
import tarfile
import urllib.request
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")


def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()


fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(
    os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(
    os.listdir(SPAM_DIR)) if len(name) > 20]


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


ham_emails = [load_email(is_spam=False, filename=name)
              for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name)
               for name in spam_filenames]

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


def email_to_text(email):
    html = None
    if (type(email) == str):
        return email
    else:
        for part in email.walk():
            ctype = part.get_content_type()
            if not ctype in ("text/plain", "text/html"):
                continue
            try:
                content = part.get_content()
            except:  # in case of encoding issues
                content = str(part.get_payload())
            if ctype == "text/plain":
                return content
            else:
                html = content
    if html:
        return html_to_plain_text(html)


try:
    import nltk

    stemmer = nltk.PorterStemmer()
except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None

try:
    import urlextract

    url_extractor = urlextract.URLExtract()
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(
                    r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index,
                            (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)
X_test_transformed = preprocess_pipeline.transform(X_test)

# log_clf = LogisticRegression(
#     solver="lbfgs", max_iter=1000, random_state=42)
# log_clf.fit(X_train_transformed, y_train)

# y_pred = log_clf.predict(X_test_transformed)
# print("Accuracy: {:.2f}%".format(
#     100 * accuracy_score(y_test, y_pred).mean()))
# print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
# print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))

rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train_transformed, y_train)
y_pred = rnd_clf.predict(X_test_transformed)
print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_pred).mean()))
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))

with open('apps/detector/spam_detector_model.pkl', 'wb') as model_file:
    pickle.dump(rnd_clf, model_file)
