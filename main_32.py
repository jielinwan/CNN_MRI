# Convolutional Neural Network
#使用3dminist

#### We use tensorFlow to build the Neural Network
import pandas as pd
from keras.utils import to_categorical
import sys
import numpy as np
from keras import regularizers
import matplotlib
import  random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
import os
import warnings
import h5py
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

###################################################
###################################################
###################################################

def Split_Train_Dev_Test(Num_batches, seed):
    "return the split id"
    np.random.seed(seed)
    #label 0
    p = np.random.permutation(Num_batches[0])
    p = p.tolist()
    num_test = Num_batches[0] // 5
    batch_train = p[0:-num_test]
    batch_dev = p[- num_test:]
    #label 1
    p = np.random.permutation(Num_batches[1])
    p1 = 2*p + Num_batches[0]
    p2=2*p+Num_batches[0]+1
    p1 = p1.tolist()
    p2=p2.tolist()

    num_test = Num_batches[1] // 5
    batch_train.extend(p1[0:-num_test])
    batch_train.extend(p2[0:-num_test])
    batch_dev.extend(p1[-num_test:])
    batch_dev.extend(p2[-num_test:])
    #label2
    p = np.random.permutation(Num_batches[2])
    p1 = 2 * p + Num_batches[0]+Num_batches[1]*2
    p2 = 2 * p + Num_batches[0] +Num_batches[1]*2+ 1
    p1 = p1.tolist()
    p2 = p2.tolist()
    num_test = Num_batches[2]  // 5
    batch_train.extend(p1[0:-num_test])
    batch_train.extend(p2[0:-num_test])
    batch_dev.extend(p1[-num_test:])
    batch_dev.extend(p2[-num_test:])
    #label3
    p = np.random.permutation(Num_batches[3])
    p1 = 2 * p + Num_batches[0] + Num_batches[1]*2+Num_batches[2]*2
    p2 = 2 * p + Num_batches[0] + Num_batches[1]*2 + Num_batches[2]*2+1
    p1 = p1.tolist()
    p2 = p2.tolist()
    num_test = Num_batches[3] // 5
    batch_train.extend(p1[0:-num_test])
    batch_train.extend(p2[0:-num_test])
    batch_dev.extend(p1[-num_test:])
    batch_dev.extend(p2[-num_test:])
    #label4
    p = np.random.permutation(Num_batches[4])
    p = p + Num_batches[0]+Num_batches[1]*2+Num_batches[2]*2+Num_batches[3]*2
    p = p.tolist()
    num_test = Num_batches[4] // 5
    batch_train.extend(p[0:-num_test])
    batch_dev.extend(p[-num_test:])

    #对源mri先分类，再根据这个结果对所有数据划分
    print(len(batch_train))

    print(sorted(batch_train))
    print(sorted(batch_dev))



    batch_train=np.array(batch_train)
    batch_train=batch_train*5
    resulttrain = batch_train.tolist()
    '''temp1=batch_train+1
    temp1=temp1.tolist()
    resulttrain.extend(temp1)'''
    temp2=batch_train+2
    temp2 = temp2.tolist()
    resulttrain.extend(temp2)
    '''temp3=batch_train+3
    temp3 = temp3.tolist()
    resulttrain.extend(temp3)'''
    temp4=batch_train+4
    temp4 = temp4.tolist()
    resulttrain.extend(temp4)


    batch_dev=np.array(batch_dev)
    batch_dev=batch_dev*5
    resulttest=batch_dev.tolist()
    '''temp = batch_dev + 1
    temp = temp.tolist()
    resulttest.extend(temp)
    temp = batch_dev + 2
    temp = temp.tolist()
    resulttest.extend(temp)
    temp = batch_dev + 3
    temp = temp.tolist()
    resulttest.extend(temp)
    temp = batch_dev + 4
    temp = temp.tolist()
    resulttest.extend(temp)'''

    print(sorted(resulttrain))
    print(sorted(resulttest))
    print(len(resulttrain))
    print(len(resulttest))
    return resulttrain, resulttest


###################################################
###################################################
###################################################
### Importing Dataset
def ImportValues(folder, batch):
    # This function loads the x and y values, reshapes X to volume and tranform Y to 1-hot

    name_img = folder + 'Img_Values' + str(batch) + '/x.npy'

    namey = folder + 'Img_Values' + str(batch) + '/y.npy'

    X_img = np.load(name_img)

    Y = np.load(namey)
    Y = np.squeeze(Y)

    # length check
    # assert(len(X_img) == len(Y))
    n_classes = 2
    X_img = np.reshape(X_img, (-1, 120, 120, 78, 1))

    # change lables to 1-hot form matrix
    # print("batch name:", namey)

    Y = to_categorical(Y, n_classes)
    Y = np.reshape(Y, (-1, 2))
    return X_img, Y

###################################################
###################################################
###################################################

def my_Scores(y_pred, y_true):
    macro = np.empty([3])
    micro = np.empty([3])
    weighted = np.empty([3])

    macro[0] = precision_score(y_true, y_pred, average='macro')
    macro[1] = recall_score(y_true, y_pred, average='macro')
    macro[2] = f1_score(y_true, y_pred, average='macro')

    micro[0] = precision_score(y_true, y_pred, average='micro')
    micro[1] = recall_score(y_true, y_pred, average='micro')
    micro[2] = f1_score(y_true, y_pred, average='micro')

    weighted[0] = precision_score(y_true, y_pred, average='weighted')
    weighted[1] = recall_score(y_true, y_pred, average='weighted')
    weighted[2] = f1_score(y_true, y_pred, average='weighted')

    return (macro, micro, weighted)


###################################################
###################################################
###################################################

### Definiton of Convolutional Newral Network

def cnn_model(x_img_input,prob1, prob2, prob3,mode, seed=None):
    regL=0.01
    xx = tf.layers.conv3d(inputs=x_img_input, filters=32, kernel_size=[3, 3, 3], padding='same',
                          kernel_regularizer=regularizers.l2(regL), name='conv1-1',
                          kernel_initializer=tf.contrib.layers.xavier_initializer())
    xx = tf.nn.relu(xx)
    #xx = tf.layers.batch_normalization(xx, training=mode)
    '''xx = tf.layers.conv3d(inputs=xx, filters=32, kernel_size=[3, 3, 3], padding='same',
                          kernel_regularizer=regularizers.l2(regL), name='conv1-2')
    xx = tf.nn.relu(xx)
    xx = tf.layers.batch_normalization(xx, training=mode)'''
    xx = tf.layers.max_pooling3d(inputs=xx, pool_size=[2, 2, 2], strides=2, name='pool1')

    xx = tf.layers.conv3d(inputs=xx, filters=64, kernel_size=[3, 3, 3], padding='same',
                         kernel_regularizer=regularizers.l2(regL), name='conv2-1')
    xx = tf.nn.relu(xx)
    #xx = tf.layers.batch_normalization(xx, training=mode)
    '''xx = tf.layers.conv3d(inputs=xx, filters=64, kernel_size=[3, 3, 3], padding='same',
                          kernel_regularizer=regularizers.l2(regL), name='conv2-2')
    xx = tf.nn.relu(xx)
    xx = tf.layers.batch_normalization(xx, training=mode)'''
    xx = tf.layers.max_pooling3d(inputs=xx, pool_size=[2, 2, 2], strides=2, name='pool2')

    xx = tf.layers.conv3d(inputs=xx, filters=128, kernel_size=[3, 3, 3], padding='same',
                         kernel_regularizer=regularizers.l2(regL), name='conv3-1')
    xx = tf.nn.relu(xx)
    #xx = tf.layers.batch_normalization(xx, training=mode)
    '''xx = tf.layers.conv3d(inputs=xx, filters=128, kernel_size=[3, 3, 3], padding='same',
                          kernel_regularizer=regularizers.l2(regL), name='conv3-2')
    xx = tf.nn.relu(xx)
    xx = tf.layers.batch_normalization(xx, training=mode)'''
    xx = tf.layers.max_pooling3d(inputs=xx, pool_size=[2, 2, 2], strides=2, name='pool3')

    '''xx = tf.layers.conv3d(inputs=xx, filters=256, kernel_size=[3, 3, 3], padding='same',
                          kernel_regularizer=regularizers.l2(regL), name='conv4-1')
    xx = tf.nn.relu(xx)
    # xx = tf.layers.batch_normalization(xx, training=mode)
    xx = tf.layers.conv3d(inputs=xx, filters=256, kernel_size=[3, 3, 3], padding='same',
                          kernel_regularizer=regularizers.l2(regL), name='conv4-2')
    xx = tf.nn.relu(xx)
    #xx = tf.layers.batch_normalization(xx, training=mode)
    xx = tf.layers.max_pooling3d(inputs=xx, pool_size=[2, 2, 2], strides=2, name='pool4')'''

    xx = tf.layers.flatten(xx)
    # 128
    xx = tf.layers.dropout(inputs=xx, rate=1 - prob1, training=mode)
    xx = tf.layers.dense(inputs=xx, units=128, kernel_regularizer=regularizers.l2(regL),
                         bias_regularizer=regularizers.l2(regL), name='dense1')
    xx = tf.nn.relu(xx)
    xx = tf.layers.dropout(inputs=xx, rate=1 - prob2, training=mode)
    xx = tf.layers.dense(inputs=xx, units=32, kernel_regularizer=regularizers.l2(regL),
                         bias_regularizer=regularizers.l2(regL), name='dense2')
    xx = tf.nn.relu(xx)
    xx = tf.layers.dropout(inputs=xx, rate=1 - prob3, training=mode)
    y_conv = tf.layers.dense(inputs=xx, units=2, name='Prediction')
    return y_conv




###################################################
###################################################
###################################################

### Training Neural Network

def train_neural_network(folder, batch_train, batch_dev, experiment_name, learning_rate, keep_rate1, keep_rate2,
                         keep_rate3, epochs):
    epoch_list = list()
    train_accu_list = list()
    train_loss_list = list()
    test_accu_list = list()
    mac_score_list = list()
    mic_score_list = list()
    weigh_score_list = list()
    tf.reset_default_graph()
    # dimensions of our input and output
    n_x = 120
    n_y = 120
    n_z = 78
    n_classes = 2
    batch_size =5
    train_mode=True
    test_mode=False

    report_Path = './Report_'+experiment_name+'/'  # XY_Values: folder for saving X matrix and Y matrix
    if not os.path.exists(report_Path):
        os.makedirs(report_Path)
    # name of the .txt file that is created
    reportFileName = 'Run_Report'

    f = open(report_Path + reportFileName + '.txt', 'w')
    f.write('\n' + 'Epoch' + '\t' + 'Train Set Accuracy' + '\t' + 'Test Set Accuracy' + '\t' + 'Elapsed time'
                                                                                               '\t' + 'mac' + '\t' + 'mic' + '\t' + 'weigh')
    f.close()
    x_img_input = tf.placeholder(tf.float32, shape=[batch_size, n_x, n_y, n_z, 1], name='Input_img')
    prob1 = tf.placeholder_with_default(1.0, shape=(), name='prob1')
    prob2 = tf.placeholder_with_default(1.0, shape=(), name='prob2')
    prob3 = tf.placeholder_with_default(1.0, shape=(), name='prob3')
    mode=tf.placeholder_with_default(True,shape=(),name='mode')
    y_input = tf.placeholder(tf.float32, shape=[batch_size, n_classes], name='Output')

    prediction = cnn_model(x_img_input, prob1, prob2, prob3, mode, seed=5)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input), name='Cost')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#这两行是为batch_normalization做全局均值计算
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate, name='Optimizer').minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32), name='accuracy')
    dev_acc_max = 0
    saver = tf.train.Saver()
    saver2 = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start_time = datetime.datetime.now()
        num_of_trn_batches = len(batch_train)//batch_size
        if(len(batch_train)%batch_size!=0):
            num_of_trn_batches+=1
        print('num_of_trn_batches:',num_of_trn_batches)

        num_of_dev_batches = len(batch_dev)//batch_size
        if(len(batch_train)%batch_size!=0):
            num_of_dev_batches+=1
        print('num_of_dev_batches:',num_of_dev_batches)
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Epoch', epoch + 1, 'started')
            epoch_train_loss = 0
            epoch_train_acc = 0
            # mini batch
            Y_pred = np.zeros((0, 0))
            Y_true = np.zeros((0, 0))

            for i_batch in range(num_of_trn_batches) :
                start_time_batch = datetime.datetime.now()

                mini_batch_x_img = []
                mini_batch_y = []
                for i in range(batch_size):
                    tag = i_batch*batch_size + i
                    if tag >= num_of_trn_batches:
                        tag = tag-num_of_trn_batches
                    x, y = ImportValues(folder, batch_train[tag])
                    mini_batch_x_img.append(x[0])
                    mini_batch_y.append(y[0])

                mini_batch_x_img = np.reshape(mini_batch_x_img, [batch_size, n_x, n_y, n_z, 1])
                mini_batch_y = np.reshape(mini_batch_y, [batch_size, n_classes])

                _, _cost,param1, y_pred, y_true = sess.run([optimizer, cost,accuracy, prediction, y_input],
                                                            feed_dict={x_img_input: mini_batch_x_img, \
                                                                       y_input: mini_batch_y,
                                                                       prob1: keep_rate1, prob2: keep_rate2,
                                                                       prob3: keep_rate3,mode:train_mode
                                                                       }
                                                            )
                epoch_train_loss += _cost
                y_pred_tip = np.argmax(y_pred, 1)
                y_true_tip = np.argmax(y_true, 1)
                print(' _cost= ' + str(_cost) + '\t _acure= ' + str(param1) + '\n y_pred= ' + str(
                    y_pred_tip) + '\n y_true=' + str(y_true_tip))

                epoch_train_acc += param1
                # Y_pred=np.append(Y_pred,y_pred)
                # Y_true=np.append(Y_true,y_true)
                end_time_batch = datetime.datetime.now()
                elapsed_time_batch = end_time_batch - start_time_batch

                print('\n', 'No. of training batches processed : ', i_batch + 1, ', Elapsed time: ', elapsed_time_batch)

                # f = open(report_Path + reportFileName + '.txt', 'a')
                # f.write('\n' + 'No. of training batches processed: ' + str(i_batch + 1) + ', Elapsed time: ' + str(elapsed_time_batch))
                # f.close()
            # mac, mic, weigh = my_Scores(Y_pred, Y_true)
            # saver.save(sess, './checkpoints/CNN_trained', global_step = epoch)
            print(epoch_train_loss)
            epoch_dev_acc = 0

            for i_batch in range(num_of_dev_batches) :
                mini_batch_x_img = []
                mini_batch_y = []
                for i in range(batch_size):
                    tag=i_batch*batch_size+i
                    if tag>=num_of_dev_batches:
                        tag=num_of_dev_batches-tag
                    x, y = ImportValues(folder, batch_dev[tag])
                    mini_batch_x_img.append(x[0])
                    mini_batch_y.append(y[0])
                mini_batch_x_img = np.reshape(mini_batch_x_img, [batch_size, n_x, n_y, n_z, 1])
                mini_batch_y = np.reshape(mini_batch_y, [batch_size, n_classes])

                param2, _ypre, _ytrue = sess.run([accuracy, prediction, y_input],
                                                 feed_dict={x_img_input: mini_batch_x_img, y_input: mini_batch_y
                                                     , prob1: keep_rate1, prob2: keep_rate2,
                                                            prob3: keep_rate3, mode: test_mode
                                                            })
                _ypre = np.argmax(_ypre, 1)
                _ytrue = np.argmax(_ytrue, 1)
                print('y_pre=' + str(_ypre), ' y_true=' + str(_ytrue))
                epoch_dev_acc += param2



            end_time_epoch = datetime.datetime.now()
            train_acc = epoch_train_acc / (num_of_trn_batches*batch_size)
            dev_acc = epoch_dev_acc / (num_of_dev_batches*batch_size)
            train_loss = epoch_train_loss / (num_of_trn_batches*batch_size)
            elapsed_time = end_time_epoch - start_time_epoch

            if dev_acc > dev_acc_max:
                dev_acc_max = dev_acc
                saver2.save(sess, report_Path + 'CNN_trained_best')

            print('\n', 'Epoch: ', epoch + 1, ', Train Set Accuracy: ' + str(train_acc) + ', Test Set Accuracy: ' +
                  str(dev_acc) + ', Elapsed time: ' + str(elapsed_time))
            f = open(report_Path + reportFileName + '.txt', 'a')
            f.write(
                '\n' + str(epoch + 1) + '\t' + str(train_acc) + '\t' + str(dev_acc) + '\t' + str(elapsed_time) + '\t')
            # str(mac) + '\t' + str(mic) + '\t' + str(weigh))
            f.close()
            epoch_list.append(epoch + 1)
            train_accu_list.append(train_acc)
            test_accu_list.append(dev_acc)
            train_loss_list.append(train_loss)

            saver.save(sess, './checkpoints/CNN_trained', global_step=epoch)
        np.save(report_Path+'train_accu_list.npy',train_accu_list)
        np.save(report_Path+'test_accu_list.npy',test_accu_list)
        np.save(report_Path+'train_loss_list.npy',train_loss_list)

        fig1 = plt.figure(1)
        plt.gcf().clear()
        ax1 = plt.gca()
        ax1.set_xlim([0, epochs + 5])
        ax1.set_ylim([0, 1.1])
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Epochs')
        plt.grid(True)

        plt.plot(epoch_list, train_accu_list, '-b', label="Training")
        plt.plot(epoch_list, test_accu_list, '-r', label="Testing")
        plt.legend(loc='upper left')

        fig2 = plt.figure(2)
        plt.gcf().clear()
        ax2 = plt.gca()
        ax2.set_xlim([0, epochs + 5])
        plt.ylabel('Loss')
        plt.xlabel('Number of Epochs')
        plt.grid(True)
        plt.plot(epoch_list, train_loss_list, '-b', label="Training")
        plt.legend(loc='upper right')

        fig1.savefig(report_Path + 'Accuracy_VS_Epoch.png', dpi=fig1.dpi)
        fig2.savefig(report_Path + 'TrainLoss_VS_Epoch.png', dpi=fig2.dpi)

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))


###################################################
###################################################
###################################################

### Example of Training

folder = 'D:/Python/DataAddNoise/ImgData/'
#experimentD:\Python\Data\DataForOnly4And0_name = str(sys.argv[0][::-1][3:][::-1])
Num_batches = [265,49,56,58,77]
seed = 0

#batch_train, batch_dev, batch_test \
all_train,all_test= Split_Train_Dev_Test(Num_batches,seed)

print(sorted(all_train))
print(sorted(all_test))
print(len(all_train))
print(len(all_test))

batch_train=all_train
#batch_train.extend(all_train)

batch_test=all_test
random.shuffle(batch_train)
random.shuffle(batch_test)
experiment_name='Final'
   #print(batch_train.shape,batch_test.shape)
train_neural_network(folder, batch_train, batch_test, experiment_name, learning_rate=0.0003, keep_rate1=0.85,
                     keep_rate2=0.85, keep_rate3=0.85, epochs=150)

