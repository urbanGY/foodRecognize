#v3 대비 incpetion layer module A version
def inception_module_A(x, input_channel, conv_1_out, conv_3_reduce_out, conv_3_out, conv_5_reduce_out, conv_5_out, pool_proj_out):
    with tf.variable_scope('inception_A') as scope:
        # 1x1 conv
        w_conv_1 = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_1_out], stddev=5e-2))
        b_conv_1 = tf.Variable(tf.constant(0.1, shape=[conv_1_out]))
        conv_1 = tf.nn.relu(tf.nn.conv2d(x, w_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1)

        # 3x3 conv
        w_conv_3_reduce = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_3_reduce_out], stddev=5e-2))
        b_conv_3_reduce = tf.Variable(tf.constant(0.1, shape=[conv_3_reduce_out]))
        conv_3_reduce = tf.nn.relu(tf.nn.conv2d(x, w_conv_3_reduce, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3_reduce)

        w_conv_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, conv_3_reduce_out, conv_3_out], stddev=5e-2))
        b_conv_3 = tf.Variable(tf.constant(0.1, shape=[conv_3_out]))
        conv_3 = tf.nn.relu(tf.nn.conv2d(conv_3_reduce, w_conv_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3)

        # 5x5 conv
        ## 1x1
        w_conv_5_reduce = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_reduce = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5_reduce = tf.nn.relu(tf.nn.conv2d(x, w_conv_5_reduce, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_reduce)

        ## 3x3
        w_conv_5_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, conv_5_reduce_out, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_1 = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5_1 = tf.nn.relu(tf.nn.conv2d(conv_5_reduce, w_conv_5_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_1)

        ## 3x3
        w_conv_5_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, conv_5_reduce_out, conv_5_out], stddev=5e-2))
        b_conv_5_2 = tf.Variable(tf.constant(0.1, shape=[conv_5_out]))
        conv_5 = tf.nn.relu(tf.nn.conv2d(conv_5_1, w_conv_5_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_2)


        # pooling
        pooling = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        w_pool = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, pool_proj_out], stddev=5e-2))
        b_pool = tf.Variable(tf.constant(0.1, shape=[pool_proj_out]))
        pool_proj = tf.nn.relu(tf.nn.conv2d(pooling, w_pool, strides=[1, 1, 1, 1], padding='SAME') + b_pool)

        concat = tf.concat([conv_1, conv_3, conv_5, pool_proj], axis=3)
        return concat


def inception_module_B(x, input_channel, conv_1_out, conv_3_reduce_out, conv_3_out, conv_5_reduce_out, conv_5_out, pool_out):
    with tf.variable_scope('inception_B') as scope: #n = 3
        #1x1 conv
        w_conv_1 = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_1_out], stddev=5e-2))
        b_conv_1 = tf.Variable(tf.constant(0.1, shape=[conv_1_out]))
        conv_1 = tf.nn.relu(tf.nn.conv2d(x, w_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1)


        #3x3 conv
        ## 1x1
        w_conv_3_reduce = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_3_reduce_out], stddev=5e-2))
        b_conv_3_reduce = tf.Variable(tf.constant(0.1, shape=[conv_3_reduce_out]))
        conv_3_reduce = tf.nn.relu(tf.nn.conv2d(x, w_conv_3_reduce, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3_reduce)

        ## 1xn
        w_conv_3_1 = tf.Variable(tf.truncated_normal(shape=[1, 3, conv_3_reduce_out, conv_3_reduce_out], stddev=5e-2))
        b_conv_3_1 = tf.Variable(tf.constant(0.1, shape=[conv_3_reduce_out]))
        conv_3_1 = tf.nn.relu(tf.nn.conv2d(conv_3_reduce, w_conv_3_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3_1)

        ## nx1
        w_conv_3_2 = tf.Variable(tf.truncated_normal(shape=[3, 1, conv_3_reduce_out, conv_3_out], stddev=5e-2))
        b_conv_3_2 = tf.Variable(tf.constant(0.1, shape=[conv_3_out]))
        conv_3 = tf.nn.relu(tf.nn.conv2d(conv_3_1, w_conv_3_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3_2)


        ##5x5 conv
        #1x1
        w_conv_5_reduce = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_reduce = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5_reduce = tf.nn.relu(tf.nn.conv2d(x, w_conv_5_reduce, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_reduce)

        ## 1xn
        w_conv_5_1 = tf.Variable(tf.truncated_normal(shape=[1, 3, conv_5_reduce_out, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_1 = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5_1 = tf.nn.relu(tf.nn.conv2d(conv_5_reduce, w_conv_5_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_1)

        ## nx1
        w_conv_5_2 = tf.Variable(tf.truncated_normal(shape=[3, 1, conv_5_reduce_out, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_2 = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5_2 = tf.nn.relu(tf.nn.conv2d(conv_5_1, w_conv_5_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_2)

        ## 1xn
        w_conv_5_3 = tf.Variable(tf.truncated_normal(shape=[1, 3, conv_5_reduce_out, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_3 = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5_3 = tf.nn.relu(tf.nn.conv2d(conv_5_2, w_conv_5_3, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_3)

        ## nx1
        w_conv_5_4 = tf.Variable(tf.truncated_normal(shape=[3, 1, conv_5_reduce_out, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_4 = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5 = tf.nn.relu(tf.nn.conv2d(conv_5_3, w_conv_5_4, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_4)


        # pooling
        pooling = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        w_pool = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, pool_out], stddev=5e-2))
        b_pool = tf.Variable(tf.constant(0.1, shape=[pool_out]))
        pool_proj = tf.nn.relu(tf.nn.conv2d(pooling, w_pool, strides=[1, 1, 1, 1], padding='SAME') + b_pool)

        concat = tf.concat([conv_1, conv_3, conv_5, pool_proj], axis=3)
        return concat


# model b, c 에서 필터 concat 하기 전에 각 연산 결과의 필터 수를 어떻게 해야하는가
def inception_module_C(x, input_channel, conv_1_out, conv_3_reduce_out, conv_3_out, conv_5_reduce_out, conv_5_out, pool_out):
    with tf.variable_scope('inception_C') as scope: #n = 3
        #1x1 conv
        w_conv_1 = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_1_out], stddev=5e-2))
        b_conv_1 = tf.Variable(tf.constant(0.1, shape=[conv_1_out]))
        conv_1 = tf.nn.relu(tf.nn.conv2d(x, w_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1)


        #3x3 conv
        ## 1x1
        w_conv_3_reduce = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_3_reduce_out], stddev=5e-2))
        b_conv_3_reduce = tf.Variable(tf.constant(0.1, shape=[conv_3_reduce_out]))
        conv_3_reduce = tf.nn.relu(tf.nn.conv2d(x, w_conv_3_reduce, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3_reduce)

        ## 1x3
        w_conv_3_1 = tf.Variable(tf.truncated_normal(shape=[1, 3, conv_3_reduce_out, conv_3_out], stddev=5e-2))
        b_conv_3_1 = tf.Variable(tf.constant(0.1, shape=[conv_3_out]))
        conv_3_1 = tf.nn.relu(tf.nn.conv2d(conv_3_reduce, w_conv_3_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3_1)

        ## 3x1
        w_conv_3_2 = tf.Variable(tf.truncated_normal(shape=[3, 1, conv_3_reduce_out, conv_3_out], stddev=5e-2))
        b_conv_3_2 = tf.Variable(tf.constant(0.1, shape=[conv_3_out]))
        conv_3_2 = tf.nn.relu(tf.nn.conv2d(conv_3_reduce, w_conv_3_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_3_2)


        ##5x5 conv
        #1x1
        w_conv_5_reduce = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, conv_5_reduce_out], stddev=5e-2))
        b_conv_5_reduce = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5_reduce = tf.nn.relu(tf.nn.conv2d(x, w_conv_5_reduce, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_reduce)

        ## 3x3
        w_conv_5 = tf.Variable(tf.truncated_normal(shape=[3, 3, conv_5_reduce_out, conv_5_reduce_out], stddev=5e-2))
        b_conv_5 = tf.Variable(tf.constant(0.1, shape=[conv_5_reduce_out]))
        conv_5 = tf.nn.relu(tf.nn.conv2d(conv_5_reduce, w_conv_5, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5)

        ## 1x3
        w_conv_5_1 = tf.Variable(tf.truncated_normal(shape=[1, 3, conv_5_reduce_out, conv_5_out], stddev=5e-2))
        b_conv_5_1 = tf.Variable(tf.constant(0.1, shape=[conv_5_out]))
        conv_5_1 = tf.nn.relu(tf.nn.conv2d(conv_5, w_conv_5_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_1)

        ## 3x1
        w_conv_5_2 = tf.Variable(tf.truncated_normal(shape=[3, 1, conv_5_reduce_out, conv_5_out], stddev=5e-2))
        b_conv_5_2 = tf.Variable(tf.constant(0.1, shape=[conv_5_out]))
        conv_5_2 = tf.nn.relu(tf.nn.conv2d(conv_5, w_conv_5_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_5_2)


        # pooling
        pooling = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        w_pool = tf.Variable(tf.truncated_normal(shape=[1, 1, input_channel, pool_out], stddev=5e-2))
        b_pool = tf.Variable(tf.constant(0.1, shape=[pool_out]))
        pool_proj = tf.nn.relu(tf.nn.conv2d(pooling, w_pool, strides=[1, 1, 1, 1], padding='SAME') + b_pool)

        concat = tf.concat([conv_1, conv_3_1, conv_3_2, conv_5_1, conv_5_2, pool_proj], axis=3)
        return concat
