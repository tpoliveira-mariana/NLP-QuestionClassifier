#!/usr/bin/env python
# coding: utf-8

# In[3]:


dev = open('DEV.txt', 'r')
questions = open('DEV-questions.txt', 'w')
labels = open('DEV-labels.txt', 'w')

for line in dev:
    treated = line.split(' ', 1)
    treated[0] += '\n'    # add \n to label
    labels.write(treated[0])
    questions.write(treated[1])

dev.close()
questions.close()
labels.close()


# In[ ]:




