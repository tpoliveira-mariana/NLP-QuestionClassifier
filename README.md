# NLP-QuestionClassifier

### Goal
Build two models that classify questions according with a coarse and a fine-grained questions’
taxonomy, from Li and Roth (see Table below).

![Coarse and fine labels](https://github.com/tpoliveira-mariana/NLP-QuestionClassifier/blob/master/Coarse%26FineLabels.png)

As an example, given the following questions:
_What fowl grabs the spotlight after the Chinese Year of the Monkey?
What is the full form of .com?
What contemptible scoundrel stole the cork from my lunch?_

Considering the coarse taxonomy (COARSE labels), your system should return:
_ENTY
ABBR
HUM_

Considering the fine taxonomy (_COARSE:fine labels_) the system should return:
_ENTY:animal
ABBR:exp
HUM:ind_

### How
There is a training set (TRAIN.txt) and a development set (DEV.txt). 
The latter is divided into:
* DEV-questions.txt that only includes the questions.
* DEV-labels.txt that only includes the labels.

During the development stage, the performance of the models is acessed as follows:
1. Predict the labels for the data:
```
python qc.py -coarse TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the coarse model
python qc.py -fine TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the fine model
```
2. Evaluate the performance of the model using the true labels:
```
python ./evaluate.py DEV-labels.txt predicted-labels.txt
```

To build the models any technique learned in NLP can be used, **except for deep learning or neural word embedings**.
