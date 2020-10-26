#!/usr/bin/env python3
# coding: utf-8

import sys
import re
import contractions
import nltk
import numpy as np
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import naive_bayes, tree
from sklearn.linear_model import SGDClassifier

STOPWORDS=[
    # negation
    'no', 'not',

    # punctuation
    ',', '?', '!', ';', '`', '&', '``',

    # misc
    'i', 'we', 'you', 'he', 'she', 'it', 'they',
    'me', 'him', 'her', 'them',
    'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves',
    'this', 'that', 'these', 'those',

    'my', 'our', 'ours', 'your', 'yours', 'their', 'theirs', 'his', 'hers', 'its',

    'can', 'should', 'must', 'shall', 'need',
]

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
    EXPR_MIDDLE_WORDS = ['the', 'von', 'van']
    def chunk_expressions(question_tokens):
        result = [question_tokens[0]]           # first word is always capital

        expression = []
        for word in question_tokens[1:]:
            if word[0].isupper() or (word in EXPR_MIDDLE_WORDS and len(expression) == 0):
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

    # matches A.B., A.B and ABC
    FEAT_ACRONYM = re.compile(r'(^| )(([A-Za-z]\.([A-Za-z]\.)+[A-Za-z]?)|([A-Z][A-Z][A-Z]+))( |$)')
    def add_custom_features(question):
        question = FEAT_ACRONYM.sub(r'\1zzzacronym\6', question)

        return question

    NT_CONTRACTION = re.compile(r" n't( |$)")
    JOIN_CONTRACTIONS = re.compile(r"([A-Za-z])( )+('s|'re|'ll|'t)( |$)")
    SPLIT_CONTRACTIONS = re.compile(r"([A-Za-z])('s|'t|'re|'ll)( |$)")
    def expand_contractions(question):
        question = NT_CONTRACTION.sub(r' not\1', question)
        question = JOIN_CONTRACTIONS.sub(r'\1\3\4', question)
        question = contractions.fix(question)
        # only possessives should remain, keep them splitted for proper lemmatizing and stuff
        question = SPLIT_CONTRACTIONS.sub(r"\1 \2\3", question)
        return question

    def preprocessOne(question):
        question = add_custom_features(question)
        question = expand_contractions(question)
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

        self.input_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2))
        train_questions_features = self.input_vectorizer.fit_transform(train_questions)

        self.model = SGDClassifier(random_state=7)
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

        self.input_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2))
        train_questions_features = self.input_vectorizer.fit_transform(train_questions)

        self.model = naive_bayes.MultinomialNB()
        self.model.fit(train_questions_features, train_labels)

    def classify(self, questions):
        questions_features = self.input_vectorizer.transform(questions)
        preds = self.model.predict(questions_features)
        return self.labels.inverse_transform(preds)

class DecisionTreeClassifier(Classifier):
    def __init__(self, train_questions, train_labels):
        Classifier.__init__(self)

        self.labels = LabelEncoder()
        train_labels = self.labels.fit_transform(train_labels)

        self.input_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2))
        train_questions_features = self.input_vectorizer.fit_transform(train_questions)

        self.model = tree.DecisionTreeClassifier()
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
            yield self.fine_models[coarse_label].classify((question,))[0]

    def classify(self, questions):
        preds_coarse = self.classify_coarse(questions)
        return self.classify_fine(questions, preds_coarse)

def without_newlines(iterable):
  return map(lambda line: line[:-1], iterable)

def main(type_, train_data_path, test_questions_path):
    with open(train_data_path, 'r') as train_f, \
         open(test_questions_path, 'r') as testQ_f:
        train_questions, train_labels = splitLabeledQuestions(without_newlines(train_f))
        test_questions = list(without_newlines(testQ_f))
    print('Training and test data loaded', file=sys.stderr)

    train_questions = preprocessQuestions(train_questions)
    test_questions = preprocessQuestions(test_questions)
    print('Training and test data preprocessed', file=sys.stderr)

    if type_ == '-coarse':
        train_labels = selectCoarseLabels(train_labels)
        model = SVMClassifier(train_questions, train_labels)
    elif type_ == '-fine':
        model = SVMClassifier(train_questions, train_labels)
    else:
        print('Unknown label kind:', type_)
        sys.exit(1)
    print('Model trained', file=sys.stderr)

    for cat in model.classify(test_questions):
        print(cat)

if __name__ == '__main__':
    if len(sys.argv) != 3+1:
        print(f'Usage: {sys.argv[0]} -{{coarse,fine}} <training data path> <test questions path>')
        sys.exit(1)
    main(*sys.argv[1:])
