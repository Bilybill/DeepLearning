from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
#our application logic will be added here

def cnn_model_fn(features,labels,mode):
    """
    Model function for cnnn
    :param features:
    :param labels:
    :param mode:
    :return:
    """
    #Input layer
    input_layer = tf.reshape(features["x"],[-1,28,28,1])

    #Convolutional Layer #1
    conv1 = tf.layers.conv2d(       #构造一个二维卷积层。采用过滤器数量，过滤内核大小，填充和激活函数作为参数。
        inputs=input_layer,
        filters = 32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    #pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)
    #使用max-pooling算法构造一个二维池化层。采用过滤器大小和步幅作为参数。

    #Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu()
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
    #Dense Layer密集层1（1,024个神经元，丢失正则化率为0.4（概率为0.4，任何给定元素在训练期间将被丢弃））
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)
    dropout=tf.layers.dropout(
        inputs=dense,rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN
    )
    #Logits Layer密集层2（10个神经元，每个数字目标类(0-9)一个）
    logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )
    predictions = {
        #Generate predictions(for PREDICT and EVAL mode)
        "classes":tf.argmax(input=logits,axis=1),
        #Add "softmax_tensor" to the graph,It is used for PREDICT and by the logging_hook
        "probabilities":tf.nn.softmax(logits,name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    #Calculate Loss(for both TRAIN and EVAL mode)
    onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
    loss = tf.loss.sparse_softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)

    #Configure the Training Op(for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    #Add evalution metrics (for EVAL mode)

    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(
            labels=labels,predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,loss=loss,eval_metric_ops=eval_metric_ops
    )



if __name__ == "__main__":
    tf.app.run()
