import random, math
## import tensorflow as tf
import torch
from torch import nn
from torch.nn import functional
import numpy as np
from functools import reduce
from operator import mul

## dtype = tf.float32
np_dtype = np.float32

class OnsetNet(nn.Module):
    def __init__(self,
                 # mode,
                 batch_size,
                 audio_context_radius,
                 audio_nbands,
                 audio_nchannels,
                 nfeats,
                 cnn_filter_shapes,
                 # cnn_init,
                 cnn_pool,
                 cnn_rnn_zack,
                 rnn_cell_type,
                 rnn_size,
                 rnn_nlayers,
                 # rnn_init,
                 rnn_nunroll,
                 rnn_keep_prob,
                 dnn_sizes,
                 # dnn_init,
                 dnn_keep_prob,
                 dnn_nonlin,
                 target_weight_strategy, # 'rect', 'last', 'pos', 'seq'
                 grad_clip,
                 opt,
                 export_feat_name=None,
                 zack_hack=0):
        super(OnsetNet,self).__init__()

        audio_context_len = audio_context_radius * 2 + 1 # タイムスタンプをどのくらい長く拾うか

        # mode = mode
        # do_cnn = len(cnn_filter_shapes) > 0
        # do_rnn = rnn_size > 0 and rnn_nlayers > 0
        # do_dnn = len(dnn_sizes) > 0

        # if not do_rnn: # default:false
        #     assert rnn_nunroll == 1

        # if cnn_rnn_zack: # default:false
        #     assert audio_context_len == 1
        #     assert zack_hack > 0 and zack_hack % 2 == 0

        # export_feat_tensors = {}

        # # Input tensors
        # feats_audio = [batch_size * rnn_nunroll, audio_context_len, audio_nbands, audio_nchannels]
        # feats_audio_nunroll = tf.placeholder(dtype, shape=[batch_size, rnn_nunroll + zack_hack, audio_context_len, audio_nbands, audio_nchannels], name='feats_audio')
        # feats_other_nunroll = tf.placeholder(dtype, shape=[batch_size, rnn_nunroll, nfeats], name='feats_other')
        # print('feats_audio: {}'.format(feats_audio_nunroll.get_shape()))
        # print('feats_other: {}'.format(feats_other_nunroll.get_shape()))
        # if mode != 'gen':
        #     targets_nunroll = tf.placeholder(dtype, shape=[batch_size, rnn_nunroll])
        #     # TODO: tf.ones acts as an overridable placeholder but this is still awkward
        #     target_weights_nunroll = tf.ones([batch_size, rnn_nunroll], dtype)

        # # Reshape input tensors to remove nunroll dim; will briefly restore later during RNN if necessary
        # if cnn_rnn_zack:
        #     feats_audio = tf.reshape(feats_audio_nunroll, shape=[batch_size, rnn_nunroll + zack_hack, audio_nbands, audio_nchannels])
        # else:
        #     feats_audio = tf.reshape(feats_audio_nunroll, shape=[batch_size * rnn_nunroll, audio_context_len, audio_nbands, audio_nchannels]) # (n,w,h,c)
        # feats_other = tf.reshape(feats_other_nunroll, shape=[batch_size * rnn_nunroll, nfeats])
        # if mode != 'gen':
        #     targets = tf.reshape(targets_nunroll, shape=[batch_size * rnn_nunroll])
        #     target_weights = tf.reshape(target_weights_nunroll, shape=[batch_size * rnn_nunroll])

        # CNN layers
        cnn_1_filter, cnn_2_filter = cnn_filter_shapes # [(band,time,channel),(band,time,channel)]
        pool_1_kernel, pool_2_kernel = cnn_pool # [(band,time),(band,time)]

        self.cnn_input_shape = [batch_size*rnn_nunroll, audio_nchannels, audio_nbands, audio_context_len] # (n,c,h,w)
        self.eval_cnn_input_shape = [1*rnn_nunroll, audio_nchannels, audio_nbands, audio_context_len]

        self.other_input_shape = [batch_size*rnn_nunroll, nfeats] # nfeats = number of other_feats

        self.cnn_1_kwargs = {'in_channels'  : audio_nchannels,
                             'out_channels' : cnn_1_filter[2],
                             'kernel_size'  : tuple(cnn_1_filter[:-1]),
                             'stride'       : 1,
                             'padding'      : 0,
                             'dilation'     : 1,   # 'VALID'
                             'bias'         : True}
        self.cnn_1_output_shape = self.calculate_cnn_output(self.cnn_input_shape, **self.cnn_1_kwargs)

        self.pool_1_kwargs = {'kernel_size' : tuple(pool_1_kernel),
                              'stride'      : tuple(pool_1_kernel),
                              'ceil_mode'   : True, # 'SAME'
                              'dilation'    : 1}
        self.pool_1_output_shape = self.calculate_pool_output(self.cnn_1_output_shape, **self.pool_1_kwargs)

        self.cnn_2_kwargs = {'in_channels'  : cnn_1_filter[2],
                             'out_channels' : cnn_2_filter[2],
                             'kernel_size'  : tuple(cnn_2_filter[:-1]),
                             'stride'       : 1,
                             'padding'      : 0, # 'VALID'
                             'dilation'     : 1,
                             'bias'         : True}
        self.cnn_2_output_shape = self.calculate_cnn_output(self.pool_1_output_shape, **self.cnn_2_kwargs)

        self.pool_2_kwargs = {'kernel_size' : tuple(pool_2_kernel),
                              'stride'      : tuple(pool_2_kernel),
                              'ceil_mode'   : True, # 'SAME'
                              'dilation'    : 1}
        self.pool_2_output_shape = self.calculate_pool_output(self.cnn_2_output_shape, **self.pool_2_kwargs)

        self.cnn_1 = nn.Conv2d(**self.cnn_1_kwargs)
        self.pool_1 = nn.MaxPool2d(**self.pool_1_kwargs)
        self.cnn_2 = nn.Conv2d(**self.cnn_2_kwargs)
        self.pool_2 = nn.MaxPool2d(**self.pool_2_kwargs)
        nn.init.uniform_(self.cnn_1.weight, *self.calculate_init_factor(self.cnn_1, factor=1.43))
        nn.init.constant_(self.cnn_1.bias, 0.1)
        nn.init.uniform_(self.cnn_2.weight, *self.calculate_init_factor(self.cnn_2, factor=1.43))
        nn.init.constant_(self.cnn_2.bias, 0.1)

        # LSTM layers
        self.lstm_input_shape_before_concat = [batch_size, rnn_nunroll, self.pool_2_output_shape[1]*self.pool_2_output_shape[2]*self.pool_2_output_shape[3]] # batch_first
        self.eval_lstm_input_shape_before_concat = [1, rnn_nunroll, self.pool_2_output_shape[1]*self.pool_2_output_shape[2]*self.pool_2_output_shape[3]] # batch_first
        self.lstm_input_shape = [batch_size, rnn_nunroll, self.pool_2_output_shape[1]*self.pool_2_output_shape[2]*self.pool_2_output_shape[3] + nfeats] # batch_first
        self.lstm_kwargs = {'input_size'    : self.lstm_input_shape[2],
                            'hidden_size'   : rnn_size,
                            'num_layers'    : 2,
                            'bias'          : True,
                            'batch_first'   : True,
                            'dropout'       : rnn_keep_prob,
                            'bidirectional' : False}
        self.lstm_output_shape = [batch_size, rnn_nunroll, rnn_size]
        self.lstm = nn.LSTM(**self.lstm_kwargs)

        # linear layers
        self.linear_input_shape = self.lstm_output_shape
        self.linear_1_kwargs = {'in_features'  : self.linear_input_shape[2],
                                'out_features' : dnn_sizes[0],
                                'bias'         : True}
        self.linear_2_kwargs = {'in_features'  : dnn_sizes[0],
                                'out_features' : dnn_sizes[1],
                                'bias'         : True}
        self.linear_output_shape = [batch_size, rnn_nunroll, dnn_sizes[1]]
        self.linear_1 = nn.Linear(**self.linear_1_kwargs)
        self.linear_2 = nn.Linear(**self.linear_2_kwargs)
        nn.init.uniform_(self.linear_1.weight, *self.calculate_init_factor(self.linear_1, factor=1.15))
        nn.init.constant_(self.linear_1.bias, 0.1)
        nn.init.uniform_(self.linear_2.weight, *self.calculate_init_factor(self.linear_2, factor=1.15))
        nn.init.constant_(self.linear_2.bias, 0.1)

        # before last sigmoid
        self.linear_last_input_shape = self.linear_output_shape
        self.linear_last_kwargs = {'in_features'  : self.linear_output_shape[2],
                                   'out_features' : 1,
                                   'bias'         : True}
        self.linear_last_output_shape = [batch_size, rnn_nunroll, 1]
        self.linear_last = nn.Linear(**self.linear_last_kwargs)
        self.init_tf_truncated_normal_(self.linear_last.weight, stddev=1./self.linear_output_shape[2])
        nn.init.constant_(self.linear_last.bias, 0.)

        self.batch_size = batch_size
        self.rnn_nunroll = rnn_nunroll
        self.zack_hack_div_2 = 0
        self.do_rnn = True
        self.target_weight_strategy = target_weight_strategy

    def forward(self, x, other, hidden=None):
        if self.training:
            x = x.reshape(self.cnn_input_shape)
        else:
            x = x.reshape(self.eval_cnn_input_shape)
        x.permute(0,3,2,1)
        x = functional.relu(self.cnn_1(x))
        x = self.pool_1(x)
        x = functional.relu(self.cnn_2(x))
        x = self.pool_2(x)
        x = torch.flatten(x, start_dim=1)    # [batch*unroll, all_feats]
        if self.training:
            x = x.reshape(self.lstm_input_shape_before_concat) # [batch, unroll, all_feats]
        else:
            x = x.reshape(self.eval_lstm_input_shape_before_concat) # [batch, unroll, all_feats]
        x = torch.cat((x,other),2)           # [batch, unroll, all_feats+other]
        x, hidden = self.lstm(x, hidden)
        x = functional.relu(self.linear_1(x))
        x = functional.relu(self.linear_2(x))
        x = self.linear_last(x)
        x = torch.squeeze(x,dim=-1)
        return x, hidden


        # cnn_output = feats_audio
        # if do_cnn:
        #     layer_last = feats_audio
        #     nfilt_last = audio_nchannels
        #     for i, ((ntime, nband, nfilt), (ptime, pband)) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
        #         layer_name = 'cnn_{}'.format(i)
        #         with tf.variable_scope(layer_name):
        #             filters = tf.get_variable('filters', [ntime, nband, nfilt_last, nfilt], initializer=cnn_init, dtype=dtype)
        #             biases = tf.get_variable('biases', [nfilt], initializer=tf.constant_initializer(0.1), dtype=dtype)
        #         if cnn_rnn_zack:
        #             padding = 'SAME'
        #         else:
        #             padding = 'VALID'

        #         conv = tf.nn.conv2d(layer_last, filters, [1, 1, 1, 1], padding=padding)
        #         biased = tf.nn.bias_add(conv, biases)
        #         convolved = tf.nn.relu(biased)

        #         pool_shape = [1, ptime, pband, 1]
        #         pooled = tf.nn.max_pool(convolved, ksize=pool_shape, strides=pool_shape, padding='SAME')
        #         print('{}: {}'.format(layer_name, pooled.get_shape()))

        #         export_feat_tensors[layer_name] = pooled

        #         # TODO: CNN dropout?

        #         layer_last = pooled
        #         nfilt_last = nfilt

        #     cnn_output = layer_last

        # # Flatten CNN and concat with other features
        # zack_hack_div_2 = 0
        # if cnn_rnn_zack:
        #     zack_hack_div_2 = zack_hack // 2
        #     cnn_output = tf.slice(cnn_output, [0, zack_hack_div_2, 0, 0], [-1, rnn_nunroll, -1, -1])
        #     nfeats_conv = reduce(lambda x, y: x * y, [int(x) for x in cnn_output.get_shape()[-2:]])
        # else:
        #     nfeats_conv = reduce(lambda x, y: x * y, [int(x) for x in cnn_output.get_shape()[-3:]])
        # feats_conv = tf.reshape(cnn_output, [batch_size * rnn_nunroll, nfeats_conv])
        # nfeats_tot = nfeats_conv + nfeats
        # feats_all = tf.concat([feats_conv, feats_other], axis=1)
        # print('feats_cnn: {}'.format(feats_conv.get_shape()))
        # print('feats_all: {}'.format(feats_all.get_shape()))

        # # Project to RNN size
        # rnn_output = feats_all
        # rnn_output_size = nfeats_tot
        # if do_rnn:
        #     with tf.variable_scope('rnn_proj'):
        #         rnn_proj_w = tf.get_variable('W', [nfeats_tot, rnn_size], initializer=tf.uniform_unit_scaling_initializer(factor=1.0, dtype=dtype), dtype=dtype)
        #         rnn_proj_b = tf.get_variable('b', [rnn_size], initializer=tf.constant_initializer(0.0), dtype=dtype)

        #     rnn_inputs = tf.nn.bias_add(tf.matmul(feats_all, rnn_proj_w), rnn_proj_b)
        #     rnn_inputs = tf.reshape(rnn_inputs, [batch_size, rnn_nunroll, rnn_size])
        #     rnn_inputs = tf.split(rnn_inputs, rnn_nunroll, axis=1)
        #     rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]

        #     if rnn_cell_type == 'rnn':
        #         cell_fn = tf.nn.rnn_cell.BasicRNNCell
        #     elif rnn_cell_type == 'gru':
        #         cell_fn = tf.nn.rnn_cell.GRUCell
        #     elif rnn_cell_type == 'lstm':
        #         cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        #     else:
        #         raise NotImplementedError()
        #     cell = cell_fn(rnn_size)

        #     if mode == 'train' and rnn_keep_prob < 1.0:
        #         cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=rnn_keep_prob)

        #     if rnn_nlayers > 1:
        #         cell = tf.nn.rnn_cell.MultiRNNCell([cell] * rnn_nlayers)

        #     initial_state = cell.zero_state(batch_size, dtype)

        #     # RNN
        #     # TODO: weight init
        #     with tf.variable_scope('rnn_unroll'):
        #         state = initial_state
        #         outputs = []
        #         for i in range(rnn_nunroll):
        #             if i > 0:
        #                 tf.get_variable_scope().reuse_variables()
        #             (cell_output, state) = cell(rnn_inputs[i], state)
        #             outputs.append(cell_output)
        #         final_state = state

        #     rnn_output = tf.reshape(tf.concat(outputs, axis=1), [batch_size * rnn_nunroll, rnn_size])
        #     rnn_output_size = rnn_size
        # print('rnn_output: {}'.format(rnn_output.get_shape()))

        # # Dense NN
        # dnn_output = rnn_output
        # dnn_output_size = rnn_output_size
        # if do_dnn:
        #     last_layer = rnn_output
        #     last_layer_size = rnn_output_size
        #     for i, layer_size in enumerate(dnn_sizes):
        #         layer_name = 'dnn_{}'.format(i)
        #         with tf.variable_scope(layer_name):
        #             dnn_w = tf.get_variable('W', shape=[last_layer_size, layer_size], initializer=dnn_init, dtype=dtype)
        #             dnn_b = tf.get_variable('b', shape=[layer_size], initializer=tf.constant_initializer(0.0), dtype=dtype)
        #         projected = tf.nn.bias_add(tf.matmul(last_layer, dnn_w), dnn_b)
        #         # TODO: argument nonlinearity, change bias to 0.1 if relu
        #         if dnn_nonlin == 'tanh':
        #             last_layer = tf.nn.tanh(projected)
        #         elif dnn_nonlin == 'sigmoid':
        #             last_layer = tf.nn.sigmoid(projected)
        #         elif dnn_nonlin == 'relu':
        #             last_layer = tf.nn.relu(projected)
        #         else:
        #             raise NotImplementedError()
        #         if mode == 'train' and dnn_keep_prob < 1.0:
        #             last_layer = tf.nn.dropout(last_layer, dnn_keep_prob)
        #         last_layer_size = layer_size
        #         print('{}: {}'.format(layer_name, last_layer.get_shape()))

        #         export_feat_tensors[layer_name] = last_layer

        #     dnn_output = last_layer
        #     dnn_output_size = last_layer_size

        # # Logistic regression
        # with tf.variable_scope('logit') as scope:
        #     logit_w = tf.get_variable('W', shape=[dnn_output_size, 1], initializer=tf.truncated_normal_initializer(stddev=1.0 / dnn_output_size, dtype=dtype), dtype=dtype)
        #     logit_b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer(0.0), dtype=dtype)
        # logits = tf.squeeze(tf.nn.bias_add(tf.matmul(dnn_output, logit_w), logit_b), squeeze_dims=[1])
        # prediction = tf.nn.sigmoid(logits)
        # prediction_inspect = tf.reshape(prediction, [batch_size, rnn_nunroll])
        # prediction_final = tf.squeeze(tf.slice(prediction_inspect, [0, rnn_nunroll - 1], [-1, 1]), squeeze_dims=[1])
        # print('logit: {}'.format(logits.get_shape()))

        # # Compute loss
        # if mode != 'gen':
        #     neg_log_lhoods = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
        #     if target_weight_strategy == 'rect':
        #         avg_neg_log_lhood = tf.reduce_mean(neg_log_lhoods)
        #     else:
        #         neg_log_lhoods = tf.multiply(neg_log_lhoods, target_weights)
        #         # be careful to have at least one weight be nonzero
        #         # should we be taking the mean elem-wise by batch? i think this is a big bug
        #         avg_neg_log_lhood = tf.reduce_sum(neg_log_lhoods) / tf.reduce_sum(target_weights)
        #     neg_log_lhoods_inspect = tf.reshape(neg_log_lhoods, [batch_size, rnn_nunroll])

        # # Train op
        # if mode == 'train':
        #     lr = tf.Variable(0.0, trainable=False)
        #     self._lr = lr
        #     self._lr_summary = tf.summary.scalar('learning_rate', self._lr)

        #     tvars = tf.trainable_variables()
        #     grads = tf.gradients(avg_neg_log_lhood, tvars)
        #     if grad_clip > 0.0:
        #         grads, _ = tf.clip_by_global_norm(grads, grad_clip)

        #     if opt == 'sgd':
        #         optimizer = tf.train.GradientDescentOptimizer(lr)
        #     else:
        #         raise NotImplementedError()

        #     train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.contrib.framework.get_or_create_global_step())

        # # Tensor exports
        # self.feats_audio = feats_audio_nunroll
        # self.feats_other = feats_other_nunroll
        # if export_feat_name:
        #     self.feats_export = export_feat_tensors[export_feat_name]
        # self.prediction = prediction_inspect
        # self.prediction_final = prediction_final
        # if mode != 'gen':
        #     self.neg_log_lhoods = neg_log_lhoods_inspect
        #     self.avg_neg_log_lhood = avg_neg_log_lhood
        #     self.targets = targets_nunroll
        #     self.target_weights = target_weights_nunroll
        # if mode == 'train':
        #     self.train_op = train_op
        # if mode != 'train' and do_rnn:
        #     self.initial_state = initial_state
        #     self.final_state = final_state
        # self.zack_hack_div_2 = zack_hack_div_2

        # self.mode = mode
        # self.batch_size = batch_size
        # self.rnn_nunroll = rnn_nunroll
        # self.do_rnn = do_rnn
        # self.target_weight_strategy = target_weight_strategy

    # def assign_lr(self, sess, lr_new):
    #     assert self.mode == 'train'
    #     sess.run(tf.assign(self._lr, lr_new))
    #     return sess.run(self._lr_summary)

    def prepare_train_batch(self, charts, randomize_charts=False, **kwargs):
        # process kwargs
        exclude_kwarg_names = ['exclude_onset_neighbors', 'exclude_pre_onsets', 'exclude_post_onsets', 'include_onsets']
        exclude_kwargs = {k:v for k,v in kwargs.items() if k in exclude_kwarg_names}
        # tn_exclusions = {
        # 'randomize_charts': args.randomize_charts,
        # 'exclude_onset_neighbors': args.exclude_onset_neighbors,
        # 'exclude_pre_onsets': args.exclude_pre_onsets,
        # 'exclude_post_onsets': args.exclude_post_onsets,
        # 'include_onsets': not args.balanced_class
        # }
        feat_kwargs = {k:v for k,v in kwargs.items() if k not in exclude_kwarg_names}
        # feats_config = {
        # 'time_context_radius': args.audio_context_radius,
        # 'diff_feet_to_id': diff_feet_to_id,
        # 'diff_coarse_to_id': diff_coarse_to_id,
        # 'diff_dipstick': args.feat_diff_dipstick,
        # 'freetext_to_id': freetext_to_id,
        # 'beat_phase': args.feat_beat_phase,
        # 'beat_phase_cos': args.feat_beat_phase_cos
        # }

        # pick random chart and sample balanced classes
        if randomize_charts: # ランダムピックにするのでこれらの設定は適用されない
            del exclude_kwargs['exclude_pre_onsets']
            del exclude_kwargs['exclude_post_onsets']
            del exclude_kwargs['include_onsets']
            if self.do_rnn:
                exclude_kwargs['nunroll'] = self.rnn_nunroll

            # create batch
            batch_feats_audio = []
            batch_feats_other = []
            batch_targets = []
            batch_target_weights = []
            for _ in range(self.batch_size):
                chart = charts[random.randint(0, len(charts) - 1)] # ランダムピックした譜面
                frame_idx = chart.sample(1, **exclude_kwargs)[0] # 譜面からランダムピックしたフレーム番号(最初の方を除く)

                subseq_start = frame_idx - (self.rnn_nunroll - 1) # 取得したいunrollの最初の番号(最初の方を除いた理由)

                if self.target_weight_strategy == 'pos' or self.target_weight_strategy == 'posbal':
                    target_sum = 0.0
                    while target_sum == 0.0:
                        audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)
                        target_sum = np.sum(target)
                        if target_sum == 0.0:
                            frame_idx = chart.sample_blanks(1, **exclude_kwargs).pop()
                            subseq_start = frame_idx - (self.rnn_nunroll - 1) # targetにtrue positiveがなかったら選び直しにするらしい
                else:
                    feat_kwargs['zack_hack_div_2'] = self.zack_hack_div_2
                    audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)

                batch_feats_audio.append(audio)
                batch_feats_other.append(other)
                batch_targets.append(target)

                if self.target_weight_strategy == 'rect':
                    weight = np.ones_like(target)
                elif self.target_weight_strategy == 'last':
                    weight = np.zeros_like(target)
                    weight[-1] = 1.0 # targetについて最後のunrollからの出力だけを誤差計算に使う
                elif self.target_weight_strategy == 'pos':
                    weight = target[:] # True Positive のみを誤差計算に使う
                elif self.target_weight_strategy == 'posbal':
                    negs = set(np.where(target == 0)[0])
                    negs_weighted = random.sample(negs, int(np.sum(target)))
                    weight = target[:]
                    weight[list(negs_weighted)] = 1.0 # True Negativeも少し誤差計算に使う
                batch_target_weights.append(weight)

            # create return arrays
            batch_feats_audio = np.array(batch_feats_audio, dtype=np_dtype)
            batch_feats_other = np.array(batch_feats_other, dtype=np_dtype)
            batch_targets = np.array(batch_targets, dtype=np_dtype)
            batch_target_weights = np.array(batch_target_weights, dtype=np_dtype)

            return batch_feats_audio, batch_feats_other, batch_targets, batch_target_weights
        else:
            # さっき:1バッチの中に__ランダムな譜面の中の__ランダムな位置のフレームを入れる
            # 　これ:1バッチの中に__ある一つの譜面の中の__ランダムな位置のフレームを入れる
            chart = charts[random.randint(0, len(charts) - 1)]
            chart_nonsets = chart.get_nonsets()
            if exclude_kwargs.get('include_onsets', False):
                npos = 0
                nneg = self.batch_size
            else:
                npos = min(self.batch_size // 2, chart_nonsets)
                nneg = self.batch_size - npos
            samples = chart.sample_onsets(npos) + chart.sample_blanks(nneg, **exclude_kwargs)
            random.shuffle(samples)

            # create batch
            batch_feats_audio = []
            batch_feats_other = []
            batch_targets = []
            batch_target_weights = []
            for frame_idx in samples:
                subseq_start = frame_idx - (self.rnn_nunroll - 1)

                if self.target_weight_strategy == 'pos' or self.target_weight_strategy == 'posbal':
                    target_sum = 0.0
                    while target_sum == 0.0:
                        audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)
                        target_sum = np.sum(target)
                        if target_sum == 0.0:
                            frame_idx = chart.sample_blanks(1, **exclude_kwargs).pop()
                            subseq_start = frame_idx - (self.rnn_nunroll - 1)
                else:
                    feat_kwargs['zack_hack_div_2'] = self.zack_hack_div_2
                    audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)

                batch_feats_audio.append(audio)
                batch_feats_other.append(other)
                batch_targets.append(target)

                if self.target_weight_strategy == 'rect':
                    weight = np.ones_like(target)
                elif self.target_weight_strategy == 'last':
                    weight = np.zeros_like(target)
                    weight[-1] = 1.0
                elif self.target_weight_strategy == 'pos':
                    weight = target[:]
                elif self.target_weight_strategy == 'posbal':
                    negs = set(np.where(target == 0)[0])
                    negs_weighted = random.sample(negs, int(np.sum(target)))
                    weight = target[:]
                    weight[list(negs_weighted)] = 1.0
                batch_target_weights.append(weight)

            # create return arrays
            batch_feats_audio = np.array(batch_feats_audio, dtype=np_dtype)
            batch_feats_other = np.array(batch_feats_other, dtype=np_dtype)
            batch_targets = np.array(batch_targets, dtype=np_dtype)
            batch_target_weights = np.array(batch_target_weights, dtype=np_dtype)

            return batch_feats_audio, batch_feats_other, batch_targets, batch_target_weights

    def iterate_eval_batches(self, eval_chart, **feat_kwargs):
        # assert self.target_weight_strategy == 'seq' #おそらく特に意味ない

        if self.do_rnn:
            subseq_len = self.rnn_nunroll
            subseq_start = -(subseq_len - 1)
        else:
            subseq_len = self.batch_size
            subseq_start = 0

        for frame_idx in range(subseq_start, eval_chart.get_nframes(), subseq_len):
            feat_kwargs['zack_hack_div_2'] = self.zack_hack_div_2
            audio, other, target = eval_chart.get_subsequence(frame_idx, subseq_len, np_dtype, **feat_kwargs)
            # print(audio.shape, other.shape, target.shape)

            weight = np.ones_like(target) # 結局rectと一緒
            mask_left = max(eval_chart.get_first_onset() - frame_idx, 0)
            mask_right = max((eval_chart.get_last_onset() + 1) - frame_idx, 0)
            weight[:mask_left] = 0.0
            weight[mask_right:] = 0.0

            if self.do_rnn:
                yield audio[np.newaxis, :], other[np.newaxis, :], target[np.newaxis, :], weight[np.newaxis, :]
            else:
                yield audio[:, np.newaxis], other[:, np.newaxis], target[:, np.newaxis], weight[:, np.newaxis]

    def calculate_cnn_output(self, input_tensor, **cnn_kwargs):
        if type(cnn_kwargs['kernel_size']) == int:
            kernel_size = (cnn_kwargs['kernel_size'], cnn_kwargs['kernel_size'])
        else:
            kernel_size = cnn_kwargs['kernel_size']
        if 'padding' in cnn_kwargs:
            if type(cnn_kwargs['padding']) == int:
                padding = (cnn_kwargs['padding'], cnn_kwargs['padding'])
            else:
                padding = cnn_kwargs['padding']
        else:
            padding = (0,0)
        if 'dilation' in cnn_kwargs:
            if type(cnn_kwargs['dilation']) == int:
                dilation = (cnn_kwargs['dilation'], cnn_kwargs['dilation'])
            else:
                dilation = cnn_kwargs['dilation']
        else:
            dilation = (1,1)
        if 'stride' in cnn_kwargs:
            if type(cnn_kwargs['stride']) == int:
                stride = (cnn_kwargs['stride'], cnn_kwargs['stride'])
            else:
                stride = cnn_kwargs['stride']
        else:
            stride = (1,1)
        if 'ceil_mode' in cnn_kwargs: # pool only
            if cnn_kwargs['ceil_mode'] is True:
                func = lambda x: math.ceil(x)
            else:
                func = lambda x: math.floor(x)
        else:
            func = lambda x: math.floor(x)

        if 'out_channels' in cnn_kwargs:
            channel_out = cnn_kwargs['out_channels']
        else:
            channel_out = input_tensor[1]

        # print(padding,dilation,stride,channel_out,input_tensor)
        height_out = func((input_tensor[2] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0] + 1)
        width_out  = func((input_tensor[3] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1] + 1)
        return [input_tensor[0], channel_out, height_out, width_out]

    def calculate_pool_output(self, input_tensor, **pool_kwargs):
        return self.calculate_cnn_output(input_tensor, **pool_kwargs)

    def calculate_init_factor(self, module, factor=1.0):
        # this is emulation of tf.uniform_unit_scaling_initializer.
        dim = None
        if type(module) == nn.Conv2d:
            dim = reduce(mul, module.weight.size()[1:])
        elif type(module) == nn.Linear:
            dim = module.weight.size()[1]
        else:
            raise NotImplementedError()
        return -1.732/dim*factor, 1.732/dim*factor
        # 1.732: sqrt(3)

    def init_tf_truncated_normal_(self, weight, mean=0., stddev=1.):
        with torch.no_grad():
            var = stddev**2
            U = torch.distributions.uniform.Uniform(mean, var)
            u = U.sample(weight.data.shape)

            a = torch.Tensor([-stddev])
            b = torch.Tensor([stddev])
            Fa = 0.5 * (1 + torch.erf(a/(2**0.5)))
            Fb = 0.5 * (1 + torch.erf(b/(2**0.5)))
            weight.data = (2**0.5)*torch.erfinv(2 *((Fb - Fa) * u + Fa) - 1)
