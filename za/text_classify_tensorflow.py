import numpy as np 
from collections import Counter
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
my_graph=tf.Graph()
with tf.Session(graph=my_graph) as sess:
    x =tf.constant([1,3,6])
    y =tf.constant([1,1,1])
    op=tf.add(x,y)
    result=sess.run(fetches=op)
    print(result)



def get_vocab(train_data,test_data):
    vocab=Counter()
    for text in train_data:
        for word in text.split(' '):
            vocab[word.lower()]+=1
    for text in test_data:
        for word in text.split(' '):
            vocab[word.lower()]+=1
    return vocab

def get_word_2_index(vocab):
    word2index={}
    for i,word in enumerate(vocab):
        word2index[word]=i
    return word2index


def get_batch(df,i,batch_size):
    batches = []
    results = []
    texts = df.data[i*batch_size:i*batch_size+batch_size]
    categories = df.target[i*batch_size:i*batch_size+batch_size]
    for text in texts:
        layer = np.zeros(total_words,dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
            
        batches.append(layer)
        
    for category in categories:
        y = np.zeros((3),dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)
            
     
    return np.array(batches),np.array(results)




categories=["comp.graphics","sci.space","rec.sport.baseball"]

newsgroups_train=fetch_20newsgroups(subset="train",categories=categories)#获取训练数据和训练目标
newsgroups_test=fetch_20newsgroups(subset="test",categories=categories)#获取

vocab=get_vocab(newsgroups_train.data,newsgroups_test.data)
#print (vocab)

word2index=get_word_2_index(vocab)

#print (word2index)

total_words=len(vocab)
print (total_words) #单词总数

#mongodb://koryakosb:cpGA1857G6aP@127.0.0.1:27017/news

n_hidden_1=100
n_hidden_2=100
n_input=total_words
n_classes=3
learning_rate=0.01
batch_size=150
training_epochs=10
display_step=1
def multilayer_perceptron(input_tensor,weights,biases):
    layer_1_multiplication=tf.matmul(input_tensor,weights['h1'])
    layer_1_addition=tf.add(layer_1_multiplication,biases['b1'])
    layer_1_activation=tf.nn.relu(layer_1_addition)
    layer_2_multiplication=tf.matmul(layer_1_activation,weights['h2'])
    layer_2_addition=tf.add(layer_2_multiplication,biases['b2'])
    layer_2_activation=tf.nn.relu(layer_2_addition)

    output_multiplication=tf.matmul(layer_2_activation,weights['out'])
    output_addition=tf.add(output_multiplication,biases['out'])

    return output_addition

weights={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
input_tensor=tf.placeholder(tf.float32,[None,n_input],name="input")
output_tensor=tf.placeholder(tf.float32,[None,n_classes],name="output")
prediction=multilayer_perceptron(input_tensor,weights,biases)

entropy_loss=tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=output_tensor)

loss=tf.reduce_mean(entropy_loss)

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(len(newsgroups_train.data)/batch_size)
        for i in range(total_batch):
            batch_x,batch_y=get_batch(newsgroups_train,i,batch_size)
            c,_=sess.run([loss,optimizer],feed_dict={input_tensor:batch_x,output_tensor:batch_y})
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_cost))


#test model
    print("optimization finished!")
    index_prediction=tf.argmax(prediction,1)
    index_correct=tf.argmax(output_tensor,1)

    correct_prediction=tf.equal(index_prediction,index_correct)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    total_test_data=len(newsgroups_test.target)

    batch_x_test,batch_y_test=get_batch(newsgroups_test,0,total_test_data)

    print("Accuracy:",accuracy.eval({input_tensor:batch_x_test,output_tensor:batch_y_test}))








