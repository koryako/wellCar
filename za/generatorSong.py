# -*- coding: utf-8 -*-
import numpy as np

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()    
    return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
      data = '\n'.join(lines)   
      file = open(filename, 'w')    
      file.write(data)    
      file.close()

# load text
def text2sample():
    raw_text = load_doc('rhyme.txt')
    print(raw_text)

    cleantokens = raw_text.split()
    raw_text = ' '.join(cleantokens)
    print(raw_text)

    # organize into sequences of characters
    length = 10
    sequences = list()
    for i in range(length, len(raw_text)):    # select sequence of tokens
         seq = raw_text[i-length:i+1]    # store
         sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    print(sequences[0])

    # save sequences to file
    out_filename = 'char_sequences.txt'
    save_doc(sequences, out_filename)


#转成one-hot编码
def ToOneHot():
    from keras.utils import to_categorical
    loadin_filename = 'char_sequences.txt'
    raw_text = load_doc(loadin_filename)
    lines = raw_text.split('\n')
    #print(lines)

    chars = sorted(list(set(raw_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    #一共十一的字母 最后一个字母为 target 样本标签，10个字母输入，输出一个字母，下一个样本移动一个字母 10to1 
    sequences = np.zeros((len(lines),11))


    for i,line in enumerate(lines):
         # integer encode line
        encoded_seq = [mapping[char] for char in line]
        temp=np.array(encoded_seq)
        sequences[i,:]=temp[0:11]

    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)
    X,y = sequences[:,:-1],sequences[:,-1]
    print(X.shape)
    sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
    print(y.shape)
    y = to_categorical(y, num_classes=vocab_size)
    print(y)
    return X,y



def computerCost(X,y,theta):
    m=len(y)
    J=0
    J=(np.transpose(X*theta-y))*(X*theta-y)/(2*m)
    return J

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(X)
    n=len(theta)
    temp=np.matrix(np.zeros((n,num_iters)))
    j_history=np.zeros((num_iters,1))
    for i in range(num_iters):
        h=np.dot(X,theta)
        temp[:,i]=theta-((alpha/m)*(np.dot(np.transpose(X),h-y)))
        theta=temp[:,i]
        j_history[i]=computerCost(X,y,theta)
        print('.')
    return theta,j_history

def featureNormaliza(X):
    X_norm=np.array(X)
    mu=np.zeros((1,X.shape[1]))
    sigma=np.zeros((1,X.shape[1]))
    mu=np.mean(X_norm,0)
    sigma=np.std(X_norm,0)
    for i in range(X.shape[1]):
        X_norm[:,i]=(X_norm[:,i]-mu[i])/sigma[i]
    return X_norm,mu,sigma

X=np.array([[1,3,3,3,3],[3,5,6,7,9]])
y=np.array([1,4,5,4,3])
#theta=np.array([0.2,0.3,0.3,2.3,1.2])
X=featureNormaliza(X)
#p=computerCost(X,y,theta)
#print(p)
if __name__=='__main__':
    z=np.random.uniform(-1,1,5)
    print (z)