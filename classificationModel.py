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
    w_words = ['what', 'when', 'which', 'who', 'where', 'why', 'whom', 'how']
    extras = [',', '?', '!', ';', '`', '&', 'I', "'s", "``"]
    stopwords = stwds.words('english')
    for word in w_words:
        stopwords.remove(word)

    stopwords += extras

    return stopwords

STOPWORDS=buildStopWords()

def getLabelsQuestions(file):
    labels = []
    questions = []

    for line in file:
        treated = line.split(' ', 1)
        labels.append(treated[0])
        questions.append(treated[1][:-1])

    file.close()
    return questions, labels

def getListFromFile(file):
    result = []
    for line in file:
        result.append(line)

    return result

def cleanQuestions(questions, stopwords=STOPWORDS):
    # expressions are joined with sym
    def groupExpression(question, sym):
        result = [question[0]]           # first word is always capital
        capital = False
        expression = []
        for word in question[1:]:
            if word[0].isupper():
                expression.append(word)
                capital = True
            else:
                capital = False
                if expression:
                    grouped = sym.join(expression)
                    result.append(grouped)

                result.append(word)
                expression = []

        if capital:
            grouped = sym.join(expression)
            result.append(grouped)

        return result

    def makeTokensLowercase(tokens):
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        return tokens

    def nltk_tag_to_wordnet_tag(nltk_tag):
        wordnet_tags = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
        if nltk_tag[0] in wordnet_tags:
            return wordnet_tags[nltk_tag[0]]

        return None

    def handleLemmatization(word, sym, lemmatizer):
        if len(word.split(sym)) == 1:     # in case it receives an expressions
            tagged = nltk.pos_tag([word])
            tag = nltk_tag_to_wordnet_tag(tagged[0][1])
            if tag is not None:
                return lemmatizer.lemmatize(word, tag)

            return word

        return ''.join(word.split(sym))    # brave|new|world -> bravenewworld

    sym = '|'
    lemmatizer = WordNetLemmatizer()

    questions_tokens = []
    for question in questions:
        text_tokens = word_tokenize(question)
        text_tokens = groupExpression(text_tokens, sym)
        text_tokens = makeTokensLowercase(text_tokens)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords]
        clean_tokens = [handleLemmatization(word, sym, lemmatizer) for word in tokens_without_sw]
        questions_tokens.append(clean_tokens)

    clean_questions = []
    for question in questions_tokens:
        clean_questions.append(' '.join(question))

    return clean_questions  

def getCoarseLabels(labels):
    coarse = []
    for label in labels:
        coarse.append(label.split(':')[0])

    return coarse


# In[2]:


def preProcessDataSet(is_coarse, trainFile, testQuestionsFile, testLabelsFile):
    train_f = open(trainFile, 'r')
    testQ_f = open(testQuestionsFile, 'r')
    testL_f = open(testLabelsFile, 'r')
    train_x, train_y = getLabelsQuestions(train_f)
    test_x = getListFromFile(testQ_f)
    test_y = getListFromFile(testL_f)

    train_x = cleanQuestions(train_x)
    test_x = cleanQuestions(test_x)

    if is_coarse:
        train_y = getCoarseLabels(train_y)
        test_y = getCoarseLabels(test_y)

    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    tfidf_vect = TfidfVectorizer(use_idf=True)
    train_x_tfidf = tfidf_vect.fit_transform(train_x)
    test_x_tfidf = tfidf_vect.transform(test_x)

    #print('train_y = ', train_y.shape)
    #print('test_y = ', test_y.shape)
    #print('train_x_tfidf = ', train_x_tfidf.shape)
    #print('test_x_tfidf = ', test_x_tfidf.shape)
    #print(tfidf_vect.vocabulary_)

    return train_x_tfidf, train_y, test_x_tfidf, test_y

def useSVM(train_x_tfidf, train_y, test_x_tfidf, test_y):
    # training phase
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_x_tfidf, train_y)

    # prediction phase
    predictions_SVM = SVM.predict(test_x_tfidf)

    # accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, test_y)*100)


def useNB(train_x_tfidf, train_y, test_x_tfidf, test_y):
    # training phase
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(train_x_tfidf, train_y)

    # prediction phase
    predictions_NB = Naive.predict(test_x_tfidf)

    # accuracy
    print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, test_y)*100)

def useModels(is_coarse, use_svm, use_nb, train, devQ, devL):
    train_x_tfidf, train_y, test_x_tfidf, test_y = preProcessDataSet(is_coarse, train, devQ, devL)
    if use_svm:
        useSVM(train_x_tfidf, train_y, test_x_tfidf, test_y)

    if use_nb:
        useNB(train_x_tfidf, train_y, test_x_tfidf, test_y) 

def questionClassification(is_coarse, use_svm=False, use_nb=True, train='TRAIN.txt', devQ='DEV-questions.txt', devL='DEV-labels.txt'):
    useModels(is_coarse, use_svm, use_nb, train, devQ, devL)



# In[3]:


#np.random.seed(500)

questionClassification(is_coarse=True, use_svm=True, use_nb=True)


# In[ ]:




