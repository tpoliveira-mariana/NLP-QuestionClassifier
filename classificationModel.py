#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
from nltk.corpus import stopwords as stwds
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm, naive_bayes

def buildStopWords():
    stopwords = stwds.words('english')

    w_words = ['what', 'when', 'which', 'who', 'where', 'why', 'whom', 'how']
    for word in w_words:
        stopwords.remove(word)

    extras = [',', '?', '!', ';', '`', '&', 'I', "'s", "``"]
    stopwords += extras

    return stopwords

STOPWORDS=buildStopWords()

LABEL_SEP = ':'

def splitLabeledQuestions(labeledQuestionList):
    labels = []
    questions = []

    for labeledQuestion in labeledQuestionList:
        parts = labeledQuestion.split(' ', 1)
        labels.append(parts[0])
        questions.append(parts[1])

    return questions, labels

def preprocessQuestions(questions, stopwords=STOPWORDS):
    SEPARATOR='|'

    # expressions are joined with SEPARATOR
    def chunk_expressions(question_tokens):
        result = [question_tokens[0]]           # first word is always capital

        expression = []
        for word in question_tokens[1:]:
            if word[0].isupper():
                expression.append(word)
            else:
                if len(expression) > 0:
                    result.append(SEPARATOR.join(expression))
                    expression = []

                result.append(word)

        if len(expression) > 0:
            result.append(SEPARATOR.join(expression))

        return result

    NTLK_TAG_TO_WORDNET_TAG = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    lemmatizer = WordNetLemmatizer()
    def lemmatize(word_or_expr):
        if SEPARATOR in word_or_expr:
            # it's an expression, remove separator (no extra lemmatization required)
            return word_or_expr.replace(SEPARATOR, '')
        else:
            # it's a regular word, lemmatize it if possible
            tagged = nltk.pos_tag([word_or_expr])
            tag = NTLK_TAG_TO_WORDNET_TAG.get(tagged[0][1])
            if tag is not None:
                return lemmatizer.lemmatize(word_or_expr, tag)

            return word_or_expr

    def preprocessOne(question):
        words = word_tokenize(question)
        words_or_exprs = chunk_expressions(words)
        words_or_exprs = map(str.lower, words_or_exprs)
        words_or_exprs = filter(lambda word: word not in stopwords, words_or_exprs)
        tokens = map(lemmatize, words_or_exprs)
        return ' '.join(tokens)

    return list(map(preprocessOne, questions))

def selectCoarseLabels(labels):
    return list(map(lambda lbl: lbl.split(LABEL_SEP)[0], labels))

class Classifier():
    def classify(self, questions):
        raise NotImplementedError

class SVMClassifier(Classifier):
    def __init__(self, train_questions, train_labels):
        Classifier.__init__(self)

        self.labels = LabelEncoder()
        train_labels = self.labels.fit_transform(train_labels)

        self.input_vectorizer = TfidfVectorizer(use_idf=True)
        train_questions_features = self.input_vectorizer.fit_transform(train_questions)

        self.model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        self.model.fit(train_questions_features, train_labels)

    def classify(self, questions):
        questions_features = self.input_vectorizer.transform(questions)
        preds = self.model.predict(questions_features)
        return self.labels.inverse_transform(preds)

class NBClassifier(Classifier):
    def __init__(self, train_questions, train_labels):
        Classifier.__init__(self)

        self.labels = LabelEncoder()
        train_labels = self.labels.fit_transform(train_labels)

        self.input_vectorizer = TfidfVectorizer(use_idf=True)
        train_questions_features = self.input_vectorizer.fit_transform(train_questions)

        self.model = naive_bayes.MultinomialNB()
        self.model.fit(train_questions_features, train_labels)

    def classify(self, questions):
        questions_features = self.input_vectorizer.transform(questions)
        preds = self.model.predict(questions_features)
        return self.labels.inverse_transform(preds)

def unzip(iterable):
    return list(zip(*iterable))

def selectCoarseCategory(questions, labels, coarse_category):
    return unzip(
        filter(
            lambda t: t[1].split(LABEL_SEP)[0] == coarse_category,
            zip(questions, labels)
        )
    )

class CompositeClassifier(Classifier):
    def __init__(self, train_questions, train_labels, InnerClassifier):
        Classifier.__init__(self)

        self.coarse_model = InnerClassifier(train_questions, selectCoarseLabels(train_labels))

        self.fine_models = dict()
        for coarse_category in set(selectCoarseLabels(train_labels)):
            questions, labels = selectCoarseCategory(train_questions, train_labels, coarse_category)
            self.fine_models[coarse_category] = InnerClassifier(questions, labels)

    def classify_coarse(self, questions):
        return self.coarse_model.classify(questions)

    def classify_fine(self, questions, coarse_labels):
        for (question, coarse_label) in zip(questions, coarse_labels):
            yield self.fine_models[coarse_label].classify((question,))

    def classify(self, questions):
        preds_coarse = self.classify_coarse(questions)
        return list(self.classify_fine(questions, preds_coarse))

#np.random.seed(500)

TRAIN_FILE = 'TRAIN.txt'
TEST_QUESTIONS_FILE = 'DEV.txt'
TEST_LABELS_FILE = 'DEV-labels.txt'

def without_newlines(iterable):
  return map(lambda line: line[:-1], iterable)

with open(TRAIN_FILE, 'r') as train_f, \
  open(TEST_QUESTIONS_FILE, 'r') as testQ_f, \
  open(TEST_LABELS_FILE, 'r') as testL_f:
    train_questions, train_labels = splitLabeledQuestions(without_newlines(train_f))
    test_questions = list(without_newlines(testQ_f))
    test_labels = list(without_newlines(testL_f))

train_labels_coarse = selectCoarseLabels(train_labels)
test_labels_coarse = selectCoarseLabels(test_labels)
train_questions = preprocessQuestions(train_questions)
test_questions = preprocessQuestions(test_questions)
print("Preprocessing complete")

svmCoarsePred = SVMClassifier(train_questions, train_labels_coarse).classify(test_questions)
print("SVM coarse Accuracy Score -> ", accuracy_score(svmCoarsePred, test_labels_coarse)*100)

nbCoarsePred = NBClassifier(train_questions, train_labels_coarse).classify(test_questions)
print("Naive Bayes coarse Accuracy Score -> ", accuracy_score(nbCoarsePred, test_labels_coarse)*100)

print()

svmFinePred = SVMClassifier(train_questions, train_labels).classify(test_questions)
print("SVM fine Accuracy Score -> ", accuracy_score(svmFinePred, test_labels)*100)

nbFinePred = NBClassifier(train_questions, train_labels).classify(test_questions)
print("Naive Bayes fine Accuracy Score -> ", accuracy_score(nbFinePred, test_labels)*100)

compositeSvmFinePred = CompositeClassifier(train_questions, train_labels, SVMClassifier).classify(test_questions)
print("Composite SVM fine Accuracy Score -> ", accuracy_score(compositeSvmFinePred, test_labels)*100)

compositeNbFinePred = CompositeClassifier(train_questions, train_labels, NBClassifier).classify(test_questions)
print("Composite Naive Bayes fine Accuracy Score -> ", accuracy_score(compositeNbFinePred, test_labels)*100)
