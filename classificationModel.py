#!/usr/bin/env python
# coding: utf-8

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

        self.model = svm.SVC(C=1.0, kernel='linear', gamma='auto')
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

def selectCategory(questions, labels, category):
    return unzip(
        filter(
            lambda t: t[1] == category,
            zip(questions, labels)
        )
    )

class CompositeClassifier(Classifier):
    def __init__(self, train_questions, train_labels, InnerCoarseClassifier, InnerFineClassifier=None):
        Classifier.__init__(self)

        if InnerFineClassifier is None:
            InnerFineClassifier = InnerCoarseClassifier

        self.coarse_model = InnerCoarseClassifier(train_questions, selectCoarseLabels(train_labels))

        self.fine_models = dict()
        for coarse_category in set(selectCoarseLabels(train_labels)):
            questions, labels = selectCoarseCategory(train_questions, train_labels, coarse_category)
            self.fine_models[coarse_category] = InnerFineClassifier(questions, labels)

    def classify_coarse(self, questions):
        return self.coarse_model.classify(questions)

    def classify_fine(self, questions, coarse_labels):
        for (question, coarse_label) in zip(questions, coarse_labels):
            yield self.fine_models[coarse_label].classify((question,))

    def classify(self, questions):
        preds_coarse = self.classify_coarse(questions)
        return list(self.classify_fine(questions, preds_coarse))

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

def dataset_stats(labels):
    coarse_freqs = dict()
    for lbl in labels:
        coarse_freqs[lbl] = coarse_freqs.get(lbl, 0) + 1

    for k, v in coarse_freqs.items():
        coarse_freqs[k] = v / len(labels)

    print(coarse_freqs)

print("Training set stats (coarse)")
dataset_stats(train_labels_coarse)

print("Training set stats (fine)")
dataset_stats(train_labels)

print("Test set stats (coarse)")
dataset_stats(test_labels_coarse)

print("Test set stats (fine)")
dataset_stats(test_labels)

train_questions = preprocessQuestions(train_questions)
test_questions = preprocessQuestions(test_questions)
print("Preprocessing complete")

#test_questions = train_questions
#test_labels_coarse = train_labels_coarse
#test_labels = train_labels
def assess_model(name, model, test_questions=test_questions, test_labels=test_labels):
    preds = model.classify(test_questions)

    overall_accuracy = accuracy_score(preds, test_labels)*100

    per_coarse_acc = dict()
    coarse_weighted_acc = 0.0
    coarse_cats = set(selectCoarseLabels(test_labels))
    for coarse_cat in coarse_cats:
        filtered_preds, labels = selectCoarseCategory(preds, test_labels, coarse_cat)

        acc = accuracy_score(filtered_preds, labels)*100
        per_coarse_acc[coarse_cat] = acc
        coarse_weighted_acc += acc
    coarse_weighted_acc /= len(coarse_cats)

    per_fine_acc = dict()
    fine_weighted_acc = 0.0
    cats = set(test_labels)
    for cat in cats:
        filtered_preds, labels = selectCategory(preds, test_labels, cat)

        acc = accuracy_score(filtered_preds, labels)*100
        per_fine_acc[cat] = acc
        fine_weighted_acc += acc
    fine_weighted_acc /= len(cats)

    print("STATS FOR MODEL:", name)
    print("Overall accuracy (%):", overall_accuracy)
    print("Coarse category-weighted accuracy (%):", coarse_weighted_acc)
    print("Fine category-weighted accuracy (%):", fine_weighted_acc)
    print("Coarse category accuracies (%):", per_coarse_acc)
    #print("Fine category accuracies (%):", per_fine_acc)
    print()


svmCoarse = SVMClassifier(train_questions, train_labels_coarse)
assess_model("SVM coarse-only", svmCoarse, test_labels=test_labels_coarse)

nbCoarse = NBClassifier(train_questions, train_labels_coarse)
assess_model("Naive Bayes", nbCoarse, test_labels=test_labels_coarse)

print("----------------------------------------------")

svmFine = SVMClassifier(train_questions, train_labels)
assess_model("SVM", svmFine)

nbFine = NBClassifier(train_questions, train_labels)
assess_model("Naive Bayes", nbFine)


compositeSvm = CompositeClassifier(train_questions, train_labels, SVMClassifier)
assess_model("Composite SVM", compositeSvm)

compositeNb = CompositeClassifier(train_questions, train_labels, NBClassifier)
assess_model("Composite Naive Bayes", compositeNb)

compositeSvmCoarseNbFine = CompositeClassifier(train_questions, train_labels, SVMClassifier, NBClassifier)
assess_model("Composite SVM Coarse/Naive Bayes Fine", compositeSvmCoarseNbFine)

