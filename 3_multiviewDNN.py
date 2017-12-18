"""
Created on Jul2017
My Implementation of the paper: End-to-End Multi-View Networks for Text Classification
@author: Yousof Erfani
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Reading data
labels = np.load('labels_50.npy')
train_data = np.load('train_data_50.npy')
# Parameters
learning_rate =.01
training_iters = 200000
batch_size = 50
display_step = 10
n_input = train_data.shape[1]
n_classes = 2
embedding_length = 50
training_epochs = 2000
split = 500
valid_data = train_data[3000:,:,:]
valid_labels = labels[3000:]
total_examples = 3000
train_data = train_data[:3000]
labels = labels[:3000]
# Place Holders
x = tf.placeholder(tf.float32, [None, n_input, embedding_length])
y = tf.placeholder(tf.float32, [None, n_classes])
keep = tf.placeholder('float')
learning_rate = tf.placeholder('float')

# First  generate 8 selections (s_i) and then generate8 views (v_i) based on those selections
def selection_generator(x, W_s, w_s, bias_Ws,bias_ws, keep):
    mult1 = tf.tensordot(x,W_s,axes =[[2],[0]]) 
# sum it with the bias
    sum1 = tf.add(mult1, bias_Ws)
# tanh activation   
    sum1= tf.nn.dropout(sum1, keep)

    tanh_select = tf.tanh(sum1) # W_s 300*200 multiply from right final : batch * max_sent*200
    mult2 = tf.tensordot(tanh_select, w_s,axes =[[2],[0]])
    m_ih = tf.add(mult2,bias_ws)
    m_ih = tf.reshape(m_ih,[-1,n_input])
    m_ih= tf.nn.dropout(m_ih, keep)

    dih = tf.nn.softmax(m_ih) # The default: softmax on the last dimension -> sentence_length
    dih = tf.layers.batch_normalization(dih)
    dih_resh = tf.reshape(dih,[-1]) 
    dih_resh = tf.tile (dih_resh, [embedding_length])
    dih_resh = tf.reshape(dih_resh,[-1,1])
    x_resh = tf.reshape(x,[-1])
    x_resh = tf.reshape(x_resh,[-1,1])
    si_resh = tf.multiply(dih_resh,x_resh)
    si_all = tf.reshape(si_resh,[-1, n_input,embedding_length])
    si = tf.reduce_sum(si_all, 1)
    return si
    
# defining the network Graph
W_s =  tf.Variable(tf.random_normal([embedding_length,200],dtype = 'float32'))
w_s =  tf.Variable(tf.random_normal([200,1]))
bias_Ws =  tf.Variable(tf.random_normal([200],dtype = 'float32'))
bias_ws =  tf.Variable(tf.random_normal([1],dtype = 'float32'))

# generate 8 selections
s1 = selection_generator(x,W_s, w_s, bias_Ws,bias_ws, keep)
s2 = selection_generator(x,W_s, w_s, bias_Ws,bias_ws, keep)
s3 = selection_generator(x,W_s, w_s, bias_Ws,bias_ws, keep)
s4 = selection_generator(x,W_s, w_s, bias_Ws,bias_ws, keep)
s5 = selection_generator(x,W_s, w_s, bias_Ws,bias_ws, keep)
s6 = selection_generator(x,W_s, w_s, bias_Ws,bias_ws, keep)
s7 = selection_generator(x,W_s, w_s, bias_Ws,bias_ws, keep)
s8 = selection_generator(x,W_s, w_s, bias_Ws,bias_ws, keep)

# generate 8 sequential views
v1 = s1
W2 =  tf.Variable(tf.random_normal([2*embedding_length,embedding_length],dtype = 'float32'))
s_concat2 = tf.concat([s1,s2],axis =1)
v2 = tf.tanh(tf.matmul(s_concat2,W2))
s_concat3 = tf.concat ([v1,v2,s3], axis=1)
W3 =  tf.Variable(tf.random_normal([3*embedding_length,embedding_length],dtype = 'float32'))
v3 = tf.tanh(tf.matmul(s_concat3,W3))
s_concat4 = tf.concat ([v1,v2,v3,s4], axis=1)
W4 =  tf.Variable(tf.random_normal([4*embedding_length,embedding_length],dtype = 'float32'))
v4 = tf.tanh(tf.matmul(s_concat4,W4))
s_concat5 = tf.concat ([v1,v2,v3,v4,s5], axis=1)
W5 =  tf.Variable(tf.random_normal([5*embedding_length,embedding_length],dtype = 'float32'))
v5 = tf.tanh(tf.matmul(s_concat5,W5))
s_concat6 = tf.concat ([v1,v2,v3,v4,v5,s6], axis=1)
W6 =  tf.Variable(tf.random_normal([6*embedding_length,embedding_length],dtype = 'float32'))
v6 = tf.tanh(tf.matmul(s_concat6,W6))
s_concat7 = tf.concat ([v1,v2,v3,v4,v5,v6,s7], axis =1)
W7 =  tf.Variable(tf.random_normal([7*embedding_length,embedding_length],dtype = 'float32'))
v7 = tf.tanh(tf.matmul(s_concat7,W7))
v8 = s8
# concatenate Views
v_concat = tf.concat([v1,v2,v3,v4,v5,v6,v7,v8], axis = 1)
v_concat = tf.nn.dropout(v_concat, keep)

# Fully connected layer
W_full =  tf.Variable(tf.random_normal([8*embedding_length,2],dtype = 'float32'))
bias_full =tf.Variable(tf.random_normal([2],dtype = 'float32'))

# drop out layer
fully_connected_layer = tf.add(tf.matmul(v_concat,W_full), bias_full)
fully_connected_layer_drop = tf.nn.dropout(fully_connected_layer, keep)
pred = tf.nn.softmax(fully_connected_layer)

# cost function for gradient descent
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected_layer_drop, labels=y))
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# tf.train.AdamOptimizer
#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#tf.train.AdadeltaOptimizer
# run a session on the tf graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
   sess.run(init)
   avg_cost_ev = []
   avg_accur_ev = []
   avg_accur_valid_ev = []
   lr=.0005
   print("Training started....")    
   for epoch in range(training_epochs):
#       lr=lr*.95
       avg_cost = 0.
       avg_acc = 0.
       acc_valid = 0.
       total_batch = int(total_examples/batch_size)
       batch_size_valid =  total_batch 
       for i in range(total_batch):
           batch_x = train_data[i*batch_size:(i+1)*batch_size]
           batch_y = labels[i*batch_size:(i+1)*batch_size] # .2
           _, c= sess.run([optimizer, cost ], feed_dict={x: batch_x,\
                                                          y: batch_y , keep: .2,learning_rate : lr})
                                                          
           acc = accuracy.eval(feed_dict={x:  batch_x ,\
                                                          y: batch_y, keep: 1.0,learning_rate : lr})                                          
                                                                     
           avg_cost  += float(c) / total_batch
           avg_acc   += float(acc) /total_batch
       c_val_total = accuracy.eval(feed_dict={x:  valid_data , y: valid_labels , keep:1.0,learning_rate : lr}) 
       avg_cost_ev.append(avg_cost)
       avg_accur_ev.append(avg_acc)
       avg_accur_valid_ev.append(c_val_total )
       print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost), "accuracy=", \
                "{:.9f}".format(avg_acc), "accuracy_valid=", \
                "{:.9f}".format(c_val_total )  ) 
  
   print("\n Optimization Finished!------------------------------------") 
   print('\n c_val_total:', c_val_total)
  # plot the curves
   plt.plot(avg_cost_ev);plt.xlabel("Iteration");plt.ylabel("Cost value")
   plt.figure()
   plt.plot(avg_accur_ev);plt.xlabel("Iteration");plt.ylabel("accuracy")
   plt.plot(avg_accur_valid_ev)
   print ('...','dropout:', keep,'embedding_length:',embedding_length,'training_epochs:',training_epochs,'total_examples:',total_examples)
  


    

