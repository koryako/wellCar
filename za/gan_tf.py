import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('data/')
img=mnist.train.images[50]
img=img.reshape((28,28))
print (img.shape)
print(tf.__version__)
def input_sensor(realimg_size,noiseimg_size):
     real_img=tf.placeholder(tf.float32,[None,realimg_size],name='real_img')
     noise_img=tf.placeholder(tf.float32,[None,noiseimg_size],name='noise_img')
     return real_img,noise_img

def generator(noise_img,n_units,out_dim,reuse=False,alpha=0.01):
     with tf.variable_scope("generator",reuse=reuse):
           hidden1=tf.nn.dense(noise_img,n_units)
           hidden1=tf.maximum(alpha*hidden1,hidden1)
           hidden1=tf.layers.dropout(hidden1,rate=0.2)
           logits=tf.layers.dense(hidden1,out_dim)
           output=tf.tanh(logits)
           return logits,output


def discriminator(img,n_units,reuse=False,alpha=0.01):
     with rf.variable_scope("discriminator",reuse=reuse):
           hidden1=tf.layers.dense(img,n_units)
           hidden1=tf.maximum(alpha*hidden1,hidden1)
           logits=tf.layers.dense(hidden1,1)
           output=tf.sigmoid(logits)
           return logits,output



img_size=mnist.train.images[0].shape[0]

noise_size=100
g_units=128
d_units=128
alpha=0.01
learning_rate=0.001
smooth=0.1

tf.reset_default_graph()
real_img,noise_img=input_sensor(img_size,noise_size)

g_logits,g_output=generator(noise_img,g_units,img_size)
d_logits_real,d_output_real=discriminator(real_img,d_units)
d_logits_fake,d_output_fake=discriminator(g_output,d_units,reuse=True)

d_loss_real=tf.reduce_mean(tf.nn.sigmid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real))*(1-smooth))
d_loss_fake=tf.reduce_mean(tf.nn.sigmid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake)))
d_loss=tf.add(d_loss_real,d_loss_fake) 
g_loss=tf.recuce_mean(tf.nn.sigmid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(g_logits))*(1-smooth))   

train_vars=tf.trainable_variables()

g_var=[var for var in train_vars if var.name.startswith("generator")]
d_var=[var for var in train_vars if var.name.startswith("discriminator")]
d_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars)
g_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars)

batch_size=64
epochs=300
n_sample=25
samples=[]
losses=[]

saver=tf.train.Saver(var_list=g_vars)

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     for e in range(epochs):
         for batch_i in range(mnist.train.num_examples//batch_size):
              batch=mnist.train.next_batch(batch_size)
              batch_images=batch[0].reshape((batch_size,784))
              batch_images=batch_images*2-1
              batch_noise=np.random.uniform(-1,1,size=(batch_size,noise_size))
              _=sess.run(d_train_opt,feed_dict={real_img:batch_images,noise_img:batch_noise})
              _=sess.run(g_train_opt,feed_dict={noise_img:batch_noise})
         train_loss_=sess.run(d_loss,feed_dict={real_img:batch_images,noise_img:batch_noise})
         train_loss_d_real=sess.run(d_loss_real,feed_dict={real_img:batch_images,noise_img:batch_noise})
         train_loss_d_fake=sess.run(d_loss_fake,feed_dict={real_img:batch_images,noise_img:batch_noise})
         train_loss_g=sess.run(g_loss,feed_dict={noise_img:batch_noise})
         sample_noise=np.random.uniform(-1,1,size=(n_sample,noise_size))
         gen_sample=sess.run(get_generator(noise_img,g_units,img_size,reuse=True),feed_dict={noise_img:sample_noise})
         samples.append(gen_samples)
         saver.save(sess,'./checkpoints/generator.ckpt')
with open('train_samples.pkl','wb') as f:
     pickle.dump(samples,f)






 
