
# coding: utf-8

# In[25]:

#import tensorflow as tf
import numpy as np
import pandas as pd
import sys
sys.path.append('~/PycharmProjects/feedforwardNet/')
import ffnet



#pth = "/Users/gvalmerbr/Downloads/Music/"
pth = "./"

#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')


# Read data:

# In[2]:

data,target,frm,it,desc = ffnet.loadData(pth)
data[data != 0] = np.log10(data[data !=0])+9
data[data<0]=0


# Set feed forward network dimensions

# In[3]:

nn_dim = [data.shape[1],800,400,target.shape[1]]
print nn_dim
opts = {'batch_size': 100,
 'display_step': 1,
 'learning_rate': 0.001,
 'training_epochs': 100}


# Build neural net

# In[4]:

ffnet.tf.reset_default_graph
saver =ffnet.tf.train.Saver()
pred, x, y, hidden = ffnet.create_ffnn(nn_dim)


# Build cost and set optimizer

# In[5]:

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#pred_do = tf.nn.dropout(pred,0.2)
cost = ffnet.tf.losses.mean_squared_error(y, pred)
#cost = ffnet.tf.metrics.mean_cosine_distance(y, pred,0)
optimizer = ffnet.tf.train.AdamOptimizer(learning_rate=opts['learning_rate']).minimize(cost)


# initialize and stack up together, pred, cost and optimizer

# In[6]:

init = ffnet.tf.global_variables_initializer()
ffnet.sess.run(init)
nnet ={'x':x,'y':y,'cost':cost,'optimizer':optimizer,'pred':pred}


# Train!

# In[20]:

opts['training_epochs'] = 300
tf = ffnet.nn_train(data,target,nnet,opts)


# In[30]:

d = data[:,:]
p = pred.eval(feed_dict={x: d})

final_cost = cost.eval(feed_dict={x: d,y:target[:,:]})
print("Cost = ",final_cost)
print("max of pred = ",p.max())


s = 'OlfModel_' + '_'.join(str(d) for d in nn_dim) + '_err='+ '%.2f' % (final_cost)+'.tf'
print "saving", s
saver.save(ffnet.sess, "./saved_model/"+ s)
print 'done'


# In[ ]:

l1 = col[0]
print l1


# In[ ]:

v =l1.read_value()


# In[ ]:

v.shape


# In[ ]:

print v


# In[ ]:

print ffnet.getSlice(data,target,10)


# In[ ]:




# In[ ]:

e =target-p
me =sum(sum(e**2))/e.shape[0]
print me


# In[ ]:

print hidden


# In[ ]:



