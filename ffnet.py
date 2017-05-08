import tensorflow as tf
import numpy as np
import pandas as pd

sess =tf.InteractiveSession()

def loadData(pth):
    data = pd.read_csv(pth + 'DATA.csv')
    target = pd.read_csv(pth + 'TARGET.csv')
    FRM = data['FRM']
    ITEM = data.columns.values[:-1]
    DESC = target.columns.values[:-1]
    data =data.drop('FRM',1).as_matrix(None)
    target =target.drop('FRM',1).as_matrix(None)
    return data,target, FRM,ITEM,DESC


def create_weight(nn_dim):
    w = {'h1': tf.Variable(tf.random_normal([nn_dim[0], nn_dim[1]]))}
    b ={'b1':tf.Variable(tf.random_normal([nn_dim[1]]))}
    for i in range(1,len(nn_dim)-1):
        w['h'+str(i+1)] = tf.Variable(tf.random_normal([nn_dim[i], nn_dim[i+1]]))
        b['b'+str(i+1)] = tf.Variable(tf.random_normal([nn_dim[i+1]]))
    return w,b

def create_layer(x,weight,bias,activation):
    layer = tf.add(tf.matmul(x, weight), bias)
    layer = tf.nn.tanh(layer)
    return layer

def create_ffnn(nn_dim):
    w,b = create_weight(nn_dim)
    layer =[tf.placeholder("float",[None, nn_dim[0]])]
    y = tf.placeholder("float", [None, nn_dim[-1]])
    for i in range(len(w)-1):
        layer.append(create_layer(layer[-1],w['h'+str(i+1)],b['b'+str(i+1)],'t'))
    out = tf.matmul(layer[-1], tf.Variable(tf.random_normal([nn_dim[-2],nn_dim[-1]])))
    return out, layer[0], y, layer

#w,b = create_weight(nn_dim)
#input =tf.placeholder("float",[None, nn_dim[0]])
#layer  =create_layer(input,w['h1'],b['b1'],'t')

def getSlice(x,y,batchsize):
    idx = np.random.randint(0,y.shape[0], batchsize)
    return x[idx, :], y[idx, :]


def nn_train(data,target,nnet,opts):
    # Initializing the variables

    x = nnet['x']
    y = nnet['y']
    optimizer = nnet['optimizer']
    cost = nnet['cost']

    # Launch the graph



    # Training cycle
    for epoch in range(opts['training_epochs']):
        avg_cost = 0.
        total_batch = int(data.shape[0]/opts['batch_size'])
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = getSlice(data,target,opts['batch_size'])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % opts['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                  "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    return tf
    # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


pred, x, y, hidden = create_ffnn([10, 5, 2])


if __name__ == '__main__':

        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    #cost = tf.losses.mean_squared_error(y, pred)
    #optimizer = tf.train.AdamOptimizer(learning_rate=opts['learning_rate']).minimize(cost)
    opts = {'learning_rate': 0.001, \
            'training_epochs': 15, \
            'batch_size': 100, \
            'display_step': 1}

    print('ffnet loaded')
