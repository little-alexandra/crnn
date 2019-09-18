import tensorflow as tf
import numpy as np,time
from local_utils import data_utils

charset = data_utils.get_charset("charset.3770.txt")


inputdata = tf.placeholder(dtype=tf.float32,
                           shape=[128,128,len(charset)],
                           name='input')
beam_decodes, beam_prob = tf.nn.ctc_beam_search_decoder(inputs=inputdata,
                                              beam_width = 1,
                                              sequence_length= np.array(128 * [128]),
                                              merge_repeated=True)

greedy_decodes, greedy_prob = tf.nn.ctc_greedy_decoder(inputs=inputdata,
                                              sequence_length= np.array(128 * [128]),
                                              merge_repeated=True)


# sequence_length: 1-D `int32` vector containing sequence lengths,having size `[batch_size]`.
# 长度是batch个，数组每个元素是sequence长度，也就是64个像素 [64,64,...64]一共batch个。
sess = tf.Session()

_input_data = np.random.random((128,128,len(charset)))
with sess.as_default():
    now = time.time()
    greedy_d,greedy_p = sess.run(
        [greedy_decodes, greedy_prob],
        feed_dict={
            inputdata: _input_data
        })
    result = data_utils.sparse_tensor_to_str(greedy_d,charset)
    print("Greedy耗时：%d秒,结果：%s" % (time.time() - now,result))

    now = time.time()
    beam_d, beam_p = sess.run(
        [beam_decodes, beam_prob],
        feed_dict={
            inputdata: _input_data
        })
    result = data_utils.sparse_tensor_to_str(beam_d, charset)
    print("BeamSearch耗时：%d秒,结果：%s" % (time.time() - now, result))
