import tensorflow as tf
import numpy as np,time
from local_utils import data_utils
from ctc import BeamSearch

charset = data_utils.get_charset("charset.3770.txt")
seq= 128
batch = 128

#[max_time, batch_size, num_classes]
inputdata = tf.placeholder(dtype=tf.float32,
                           shape=[seq,batch,len(charset)],
                           name='input')

beam_decodes, beam_prob = tf.nn.ctc_beam_search_decoder(inputs=inputdata,
                                              beam_width = 1,
                                              sequence_length= np.array(batch * [seq]),
                                              merge_repeated=True)

greedy_decodes, greedy_prob = tf.nn.ctc_greedy_decoder(inputs=inputdata,
                                              sequence_length= np.array(batch * [seq]),
                                              merge_repeated=True)

softmax_result = tf.nn.softmax(inputdata,axis=2)
max_index = tf.argmax(softmax_result,axis=2)
print("argmax index:",max_index)
max_index_squeeze = tf.squeeze(max_index)
softmax_transport = tf.transpose(softmax_result,[1,0,2])
print("softmax_transport:",softmax_transport)
# probs     = tf.gather(softmax_transport,max_index_squeeze,axis=2)


probs = tf.reduce_max(softmax_transport,axis=2)
print("probs:",probs)

# shape = t1.shape.as_list()
# xy_ind = np.stack(np.mgrid[:shape[0], :shape[1]], axis=-1)
# gather_ind = tf.concat([xy_ind, max_ind[..., None]], axis=-1)
# sliced_t2 = tf.gather_nd(t2, gather_ind)

# data [seq,batch]
def get_string(data,charset):
    print("输入的索引：",data.shape)
    data = np.transpose(data,[1,0])
    result = []

    for one_line in data:
        values = [charset[id] for id in one_line]
        result.append(''.join(c for c in values if c != '\n'))
    return result


# sequence_length: 1-D `int32` vector containing sequence lengths,having size `[batch_size]`.
# 长度是batch个，数组每个元素是sequence长度，也就是64个像素 [64,64,...64]一共batch个。
sess = tf.Session()

_input_data = np.random.random((seq,batch,len(charset)))
with sess.as_default():

    # 1.tf自带的贪心法
    now = time.time()
    greedy_d,greedy_p = sess.run(
        [greedy_decodes, greedy_prob],
        feed_dict={
            inputdata: _input_data
        })
    result = data_utils.sparse_tensor_to_str(greedy_d[0],charset)
    print("Greedy耗时：%d秒,结果：\n%r" % (time.time() - now,result))
    # print(np.log(greedy_p))

    # 2.用beam_width=1的beam_search_decoder
    now = time.time()
    beam_d, beam_p = sess.run(
        [beam_decodes, beam_prob],
        feed_dict={
            inputdata: _input_data
        })
    result = data_utils.sparse_tensor_to_str(beam_d[0], charset)
    print("BeamSearch耗时：%d秒,结果：\n%r" % (time.time() - now, result))
    # print(np.log(np.array(beam_p)))

    # 3.自己实现的一个贪心法
    now = time.time()
    max_i,p = sess.run(
        [max_index,probs],
        feed_dict={
            inputdata: _input_data
        })
    print("自己实现,输入：", max_i.shape)
    # print("自己实现,概率：", p)
    result = get_string(max_i, charset)
    print("自己实现，耗时：%d秒,结果：\n%r" % (time.time() - now, result))

    # 4.第三方的一个ctc inference实现
    # now = time.time()
    # result = sess.run(
    #     softmax_transport,
    #     feed_dict={
    #         inputdata: _input_data
    #     })
    # chars = "".join(charset)[:-1]
    # print("BeamSearch,输出：", result.shape)
    # ss = []
    # for r in result:
    #     s = BeamSearch.ctcBeamSearch(r,chars,None,1)
    #     ss.append(s)
    # print("BeamSearch,耗时：%d秒,结果(%d)：\n%r" % (time.time() - now, len(ss[0]),ss))

