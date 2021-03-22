import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import randint
xrange = range

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope=None):
    # upsampling step in fully convolution neural network
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02), name=scope)
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def dncnn(input, is_training=True, output_channels=3):
    '''
        Model dncnn:
            + using 17 convolution layers continuously 
        parameters:
            input : a jpeg image
            output: a retore image
        
    '''

    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in xrange(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output

def FCN(input):
    '''
        Purpose: Estimate jpeg qualtity level
        the architecture consists 5 convoluion layers continuously
        parameters:
            input : jpeg image
            output: a quality jpeg level mask for original image 
    '''
    with tf.variable_scope('fcn'):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv1')
        conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv2')
        conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv3')
        conv4 = slim.conv2d(conv3, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv4')
        conv5 = slim.conv2d(conv4, 3, [3, 3], rate=1, activation_fn=lrelu, scope='conv5')
        return conv5


def UNet(input):
    '''
        Purpose: Reconstruct image 
        parameters:
            input: jpeg image and a quality jpeg level mask for original image
            output: a jpeg-deblocking image 

    '''
    with tf.variable_scope('unet'):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv1_1', weights_initializer = 'he_normal')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv1_2', weights_initializer = 'he_normal')
        pool1 = slim.avg_pool2d(conv1, [2, 2], padding='SAME')
        
        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='conv2_1', weights_initializer = 'he_normal')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='conv2_2', weights_initializer = 'he_normal')
        pool2 = slim.avg_pool2d(conv2, [2, 2], padding='SAME')


        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='conv3_1', weights_initializer = 'he_normal')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='conv3_2', weights_initializer = 'he_normal')
        pool3 = slim.avg_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='conv4_1', weights_initializer = 'he_normal')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='conv4_2', weights_initializer = 'he_normal')
        pool4 = slim.avg_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='conv5_1', weights_initializer = 'he_normal')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='conv5_2', weights_initializer = 'he_normal')

        up6 = upsample_and_concat(conv5, conv4, 256, 512, scope='deconv6')
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='conv6_1', weights_initializer = 'he_normal')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='conv6_2', weights_initializer = 'he_normal')

        up7 = upsample_and_concat(conv6, conv3, 128, 256, scope='deconv7')
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='conv7_1', weights_initializer = 'he_normal')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='conv7_2', weights_initializer = 'he_normal')

        up8 = upsample_and_concat(conv7, conv2, 64, 128, scope='deconv8')
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='conv8_1', weights_initializer = 'he_normal')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='conv8_2', weights_initializer = 'he_normal')

        up9 = upsample_and_concat(conv8, conv1, 32, 64, scope='deconv9')
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv9_1', weights_initializer = 'he_normal')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv9_2', weights_initializer = 'he_normal')

        out = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='conv10', weights_initializer = 'he_normal')
        return out

def CBDNet(input):
    '''
        Convolutional Blind Denosing model
    '''

    noise_img = FCN(input)
    concat_img = tf.concat([input, noise_img], 3)
    out = UNet(concat_img)
    return noise_img

if __name__ == "__main__":
    x = tf.placeholder( dtype = tf.float32, shape= [None , 512, 512, 1])
    CBDNet(x)