import tensorflow as tf
import numpy as np,time

inputdata = tf.placeholder(dtype=tf.float32,
                           shape=[128,65,3862],
                           name='input')
# decodes, prob = tf.nn.ctc_beam_search_decoder(inputs=inputdata,
#                                               beam_width = 1,
#                                               sequence_length= np.array(65 * [128]),
#                                               merge_repeated=False)

decodes, prob = tf.nn.ctc_greedy_decoder(inputs=inputdata,
                                              sequence_length= np.array(65 * [128]),
                                              merge_repeated=False)


# sequence_length: 1-D `int32` vector containing sequence lengths,having size `[batch_size]`.
# 长度是batch个，数组每个元素是sequence长度，也就是64个像素 [64,64,...64]一共batch个。
sess = tf.Session()

_input_data = np.random.random((128,65,3862))
with sess.as_default():
    now = time.time()
    preds, _prob = sess.run(
        [decodes, prob],
        feed_dict={
            inputdata: _input_data
        })

    print("耗时：%d" % (time.time() - now))
