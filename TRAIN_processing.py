#!/usr/bin/env python
# coding: utf-8

# In[94]:


# importing NLTK libarary stopwords 
import nltk
from nltk.corpus import stopwords as stwds
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

w_words = ['what', 'when', 'which', 'who', 'where', 'why', 'whom', 'how']
extras = [',', '?', '!', ';', '`', '&', 'I', "'s", "``"]
stopwords = stwds.words('english')
for word in w_words:
    stopwords.remove(word)
    
stopwords += extras
print(stopwords)


# In[95]:


# Separate labes from questions
train = open('TRAIN.txt', 'r');
labels = []
questions = []

for line in train:
    treated = line.split(' ', 1)
    labels.append(treated[0])
    questions.append(treated[1][:-1])

train.close()    

print(questions)


# In[96]:


# Remove stopwords and group expressions
def groupExpression(question):
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
                grouped = ' '.join(expression)
                result.append(grouped)
                
            result.append(word)
            expression = []
    
    if capital:
        grouped = ' '.join(expression)
        result.append(grouped)
    
    return result

def makeTokensLowercase(tokens):
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        
    return tokens
        


questions_tokens = []
for question in questions:
    text_tokens = word_tokenize(question)
    text_tokens = groupExpression(text_tokens)
    text_tokens = makeTokensLowercase(text_tokens)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    questions_tokens.append(tokens_without_sw)
    
print(questions_tokens)


# In[97]:


# build list with every word

words = []
for question in questions_tokens:
    for word in question:
        if not word in words:
            words.append(word)
            
print(words)


# In[ ]:




