import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.autograd as autograd
data=[("me gusta comeer en lacafeteria".split(),"spanish"),
      ("Give it to me".split(),"english"),
      ("No creo que sea una buena idea".split(),"spanish"),
      ("No it is not a good idea to get lost at sea".split(),"english")
     ]

test=[("Yo creo que is".split(),"spanish"),
      ("it is lost on me".split(),"english")]


word2index={}
for sent,_ in data+test:
   for word in sent:
       if word not in word2index:
           word2index[word]=len(word2index)


print (word2index)

vocab_size=len(word2index)
num_labels=2
class BoWClassifier(nn.Module):
    def __init__(self,num_labels,vocab_size):
        super(BoWClassifier,self).__init__()
        self.linear=nn.Linear(vocab_size,num_labels)
    def forward(self,bow_vec):
        return F.log_softmax(self.linear(bow_vec))


def make_bow_vector(sentence,word2index):
    vec=torch.zeros(len(word2index))
    for word in sentence:
        vec[word2index[word]]+=1
    return vec.view(1,-1)
    
def make_target(label,label2index):
    return torch.LongTensor([label_to_ix[label]])


model=BoWClassifier(num_labels,vocab_size)

for param in model.parameters():
     print ('param:',param)


sample=data[0]
bow_vector=make_bow_vector(sample[0],word2index)
log_probs=model(autograd.Variable(bow_vector))
print("log_probs",log_probs)

