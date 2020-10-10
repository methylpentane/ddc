from collections import defaultdict
import pickle
import os
import time
import argparse

import torch
from torch import optim
from torch import nn
import numpy as np
# import tensorflow as tf

from torch_sym_net import SymNet
from util import *

parser = argparse.ArgumentParser()
# Data
parser.add_argument('--train_txt_fp', type=str, default='', help='Training dataset txt file with a list of pickled song files')
parser.add_argument('--valid_txt_fp', type=str, default='', help='Eval dataset txt file with a list of pickled song files')
parser.add_argument('--test_txt_fp', type=str, default='', help='Test dataset txt file with a list of pickled song files')
parser.add_argument('--sym_rnn_pretrain_model_ckpt_fp', type=str, default='', help='File path to model checkpoint with only sym weights')
parser.add_argument('--model_ckpt_fp', type=str, default='', help='File path to model checkpoint if resuming or eval')

# Features
parser.add_argument('--sym_in_type', type=str, default='onehot', help='Either \'onehot\' or \'bagofarrows\'')
parser.add_argument('--sym_out_type', type=str, default='onehot', help='Either \'onehot\' or \'bagofarrows\'')
parser.add_argument('--sym_narrows', type=int, default=4, help='Number or arrows in data')
parser.add_argument('--sym_narrowclasses', type=int, default=4, help='Number or arrow classes in data')
parser.add_argument('--sym_embedding_size', type=int, default=32, help='')
parser.add_argument('--audio_z_score', action='store_true', default=False, help='If true, train and test on z-score of validation data')
parser.add_argument('--audio_deviation_max', type=int, default=0, help='')
parser.add_argument('--audio_context_radius', type=int, default=-1, help='Past and future context per training example')
parser.add_argument('--audio_nbands', type=int, default=0, help='Number of bands per frame')
parser.add_argument('--audio_nchannels', type=int, default=0, help='Number of channels per frame')
parser.add_argument('--feat_meas_phase', action='store_true', default=False, help='')
parser.add_argument('--feat_meas_phase_cos', action='store_true', default=False, help='')
parser.add_argument('--feat_meas_phase_sin', action='store_true', default=False, help='')
parser.add_argument('--feat_beat_phase', action='store_true', default=False, help='')
parser.add_argument('--feat_beat_phase_cos', action='store_true', default=False, help='')
parser.add_argument('--feat_beat_phase_sin', action='store_true', default=False, help='')
parser.add_argument('--feat_beat_diff', action='store_true', default=False, help='')
parser.add_argument('--feat_beat_diff_next', action='store_true', default=False, help='')
parser.add_argument('--feat_beat_abs', action='store_true', default=False, help='')
parser.add_argument('--feat_time_diff', action='store_true', default=False, help='')
parser.add_argument('--feat_time_diff_next', action='store_true', default=False, help='')
parser.add_argument('--feat_time_abs', action='store_true', default=False, help='')
parser.add_argument('--feat_prog_diff', action='store_true', default=False, help='')
parser.add_argument('--feat_prog_abs', action='store_true', default=False, help='')
parser.add_argument('--feat_diff_feet', action='store_true', default=False, help='')
parser.add_argument('--feat_diff_aps', action='store_true', default=False, help='')
parser.add_argument('--feat_beat_phase_nquant', type=int, default=0, help='')
parser.add_argument('--feat_beat_phase_max_nwraps', type=int, default=0, help='')
parser.add_argument('--feat_meas_phase_nquant', type=int, default=0, help='')
parser.add_argument('--feat_meas_phase_max_nwraps', type=int, default=0, help='')
parser.add_argument('--feat_diff_feet_to_id_fp', type=str, default='', help='')
parser.add_argument('--feat_diff_coarse_to_id_fp', type=str, default='', help='')
parser.add_argument('--feat_diff_dipstick', action='store_true', default=False, help='')
parser.add_argument('--feat_freetext_to_id_fp', type=str, default='', help='')
# parser.add_argument('--feat_bucket_beat_diff_n', type=int, default=None, help='')
# parser.add_argument('--feat_bucket_beat_diff_max', type=float, default=None, help='')
# parser.add_argument('--feat_bucket_time_diff_n', type=int, default=None, help='')
# parser.add_argument('--feat_bucket_time_diff_max', type=float, default=None, help='')
parser.add_argument('--feat_bucket_beat_diff_n', type=int, default=0, help='')
parser.add_argument('--feat_bucket_beat_diff_max', type=float, default=0, help='')
parser.add_argument('--feat_bucket_time_diff_n', type=int, default=0, help='')
parser.add_argument('--feat_bucket_time_diff_max', type=float, default=0, help='')

# Network params
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--nunroll', type=int, default=1, help='')
parser.add_argument('--cnn_filter_shapes', type=str, default='', help='CSV 3-tuples of filter shapes (time, freq, n)')
parser.add_argument('--cnn_pool', type=str, default='', help='CSV 2-tuples of pool amounts (time, freq)')
parser.add_argument('--cnn_dim_reduction_size', type=int, default=-1, help='')
parser.add_argument('--cnn_dim_reduction_keep_prob', type=float, default=1.0, help='')
parser.add_argument('--cnn_dim_reduction_nonlin', type=str, default='', help='')
parser.add_argument('--rnn_cell_type', type=str, default='lstm', help='')
parser.add_argument('--rnn_size', type=int, default=0, help='')
parser.add_argument('--rnn_nlayers', type=int, default=0, help='')
parser.add_argument('--rnn_keep_prob', type=float, default=1.0, help='')
parser.add_argument('--dnn_sizes', type=str, default='', help='CSV sizes for dense layers')
parser.add_argument('--dnn_keep_prob', type=float, default=1.0, help='')

# Training params
parser.add_argument('--grad_clip', type=float, default=0.0, help='Clip gradients to this value if greater than 0')
parser.add_argument('--opt', type=str, default='sgd', help='One of \'sgd\'')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=1.0, help='Multiply learning rate by this value every epoch')
parser.add_argument('--lr_decay_delay', type=int, default=0, help='')
parser.add_argument('--nbatches_per_ckpt', type=int, default=100, help='Save model weights every N batches')
parser.add_argument('--nbatches_per_eval', type=int, default=10000, help='Evaluate model every N batches')
parser.add_argument('--nepochs', type=int, default=0, help='Number of training epochs, negative means train continuously')
parser.add_argument('--experiment_dir', type=str, default='', help='Directory for temporary training files and model weights')

# Eval params

# Generate params
parser.add_argument('--generate_fp', type=str, default='', help='')
parser.add_argument('--generate_vocab_fp', type=str, default='', help='')

args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:',end='')
print(DEVICE)
# dtype = tf.float32


########################################### [main]
def main():
    assert args.experiment_dir
    do_train = args.nepochs != 0 and bool(args.train_txt_fp)
    do_valid = bool(args.valid_txt_fp)
    do_train_eval = do_train and do_valid
    do_eval = bool(args.test_txt_fp)
    do_generate = bool(args.generate_fp)

    # Load data
    print('Loading data')
    train_data, valid_data, test_data = open_dataset_fps(args.train_txt_fp, args.valid_txt_fp, args.test_txt_fp)

    # Calculate validation metrics
    if args.audio_z_score:
        z_score_fp = os.path.join(args.experiment_dir, 'valid_mean_std.pkl')
        if do_valid and not os.path.exists(z_score_fp):
            print('Calculating validation metrics')
            mean_per_band, std_per_band = calc_mean_std_per_band(valid_data)
            with open(z_score_fp, 'wb') as f:
                pickle.dump((mean_per_band, std_per_band), f)
        else:
            print('Loading validation metrics')
            with open(z_score_fp, 'rb') as f:
                mean_per_band, std_per_band = pickle.load(f)

        # Sanitize data
        for data in [train_data, valid_data, test_data]:
            apply_z_norm(data, mean_per_band, std_per_band)

    # Flatten the data into chart references for easier iteration
    print('Flattening datasets into charts')
    charts_train = flatten_dataset_to_charts(train_data)
    charts_valid = flatten_dataset_to_charts(valid_data)
    charts_test = flatten_dataset_to_charts(test_data)

    # Filter charts that are too short
    charts_train_len = len(charts_train)
    charts_train = list(filter(lambda x: x.get_nannotations() >= args.nunroll, charts_train))
    if len(charts_train) != charts_train_len:
        print('{} charts too small for training'.format(charts_train_len - len(charts_train)))
    print('Train set: {} charts, valid set: {} charts, test set: {} charts'.format(len(charts_train), len(charts_valid), len(charts_test)))

    # Load ID maps
    diff_feet_to_id = None
    if args.feat_diff_feet_to_id_fp:
        diff_feet_to_id = load_id_dict(args.feat_diff_feet_to_id_fp)
    diff_coarse_to_id = None
    if args.feat_diff_coarse_to_id_fp:
        diff_coarse_to_id = load_id_dict(args.feat_diff_coarse_to_id_fp)
    freetext_to_id = None
    if args.feat_freetext_to_id_fp:
        freetext_to_id = load_id_dict(args.feat_freetext_to_id_fp)

    # Create feature config
    feats_config = {
        'meas_phase': args.feat_meas_phase,
        'meas_phase_cos': args.feat_meas_phase_cos,
        'meas_phase_sin': args.feat_meas_phase_sin,
        'beat_phase': args.feat_beat_phase,
        'beat_phase_cos': args.feat_beat_phase_cos,
        'beat_phase_sin': args.feat_beat_phase_sin,
        'beat_diff': args.feat_beat_diff,
        'beat_diff_next': args.feat_beat_diff_next,
        'beat_abs': args.feat_beat_abs,
        'time_diff': args.feat_time_diff,
        'time_diff_next': args.feat_time_diff_next,
        'time_abs': args.feat_time_abs,
        'prog_diff': args.feat_prog_diff,
        'prog_abs': args.feat_prog_abs,
        'diff_feet': args.feat_diff_feet,
        'diff_aps': args.feat_diff_aps,
        'beat_phase_nquant': args.feat_beat_phase_nquant,
        'beat_phase_max_nwraps': args.feat_beat_phase_max_nwraps,
        'meas_phase_nquant': args.feat_meas_phase_nquant,
        'meas_phase_max_nwraps': args.feat_meas_phase_max_nwraps,
        'diff_feet_to_id': diff_feet_to_id,
        'diff_coarse_to_id': diff_coarse_to_id,
        'freetext_to_id': freetext_to_id,
        'bucket_beat_diff_n': args.feat_bucket_beat_diff_n,
        'bucket_time_diff_n': args.feat_bucket_time_diff_n
    }
    nfeats = 0
    for feat in feats_config.values():
        if feat is None:
            continue
        if isinstance(feat, dict):
            nfeats += max(feat.values()) + 1
        else:
            nfeats += int(feat)
    nfeats += 1 if args.feat_beat_phase_max_nwraps > 0 else 0
    nfeats += 1 if args.feat_meas_phase_max_nwraps > 0 else 0
    nfeats += 1 if args.feat_bucket_beat_diff_n > 0 else 0
    nfeats += 1 if args.feat_bucket_time_diff_n > 0 else 0
    feats_config['diff_dipstick'] = args.feat_diff_dipstick
    feats_config['audio_time_context_radius'] = args.audio_context_radius
    feats_config['audio_deviation_max'] = args.audio_deviation_max
    feats_config['bucket_beat_diff_max'] = args.feat_bucket_beat_diff_max
    feats_config['bucket_time_diff_max'] = args.feat_bucket_time_diff_max
    feats_config_eval = dict(feats_config)
    feats_config_eval['audio_deviation_max'] = 0
    print('Feature configuration (nfeats={}): {}'.format(nfeats, feats_config))

    # Create model config
    # rnn_proj_init = tf.constant_initializer(0.0, dtype=dtype) if args.sym_rnn_pretrain_model_ckpt_fp else tf.uniform_unit_scaling_initializer(factor=1.0, dtype=dtype)
    model_config = {
        'nunroll': args.nunroll,
        'sym_in_type': args.sym_in_type,
        'sym_embedding_size': args.sym_embedding_size,
        'sym_out_type': args.sym_out_type,
        'sym_narrows': args.sym_narrows,
        'sym_narrowclasses': args.sym_narrowclasses,
        'other_nfeats': nfeats,
        'audio_context_radius': args.audio_context_radius,
        'audio_nbands': args.audio_nbands,
        'audio_nchannels': args.audio_nchannels,
        'cnn_filter_shapes': stride_csv_arg_list(args.cnn_filter_shapes, 3, int),
        # 'cnn_init': tf.uniform_unit_scaling_initializer(factor=1.43, dtype=dtype),
        'cnn_pool': stride_csv_arg_list(args.cnn_pool, 2, int),
        'cnn_dim_reduction_size': args.cnn_dim_reduction_size,
        # 'cnn_dim_reduction_init': tf.uniform_unit_scaling_initializer(factor=1.0, dtype=dtype),
        'cnn_dim_reduction_nonlin': args.cnn_dim_reduction_nonlin,
        'cnn_dim_reduction_keep_prob': args.cnn_dim_reduction_keep_prob,
        # 'rnn_proj_init': rnn_proj_init,
        'rnn_cell_type': args.rnn_cell_type,
        'rnn_size': args.rnn_size,
        'rnn_nlayers': args.rnn_nlayers,
        # 'rnn_init': tf.random_uniform_initializer(-5e-2, 5e-2, dtype=dtype),
        'nunroll': args.nunroll,
        'rnn_keep_prob': args.rnn_keep_prob,
        'dnn_sizes': stride_csv_arg_list(args.dnn_sizes, 1, int),
        # 'dnn_init': tf.uniform_unit_scaling_initializer(factor=1.15, dtype=dtype),
        'dnn_keep_prob': args.dnn_keep_prob,
        'grad_clip': args.grad_clip,
        'opt': args.opt,
        'batch_size': args.batch_size
    }
    print('Model configuration: {}'.format(model_config))

    # torch
    print('Creating model')
    model = SymNet(**model_config).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if do_train:
        # Calculate epoch stuff
        train_nexamples = sum([chart.get_nannotations() for chart in charts_train])
        examples_per_batch = args.batch_size
        examples_per_batch *= model.out_nunroll
        batches_per_epoch = train_nexamples // examples_per_batch
        nbatches = args.nepochs * batches_per_epoch
        print('{} frames in data, {} batches per epoch, {} batches total'.format(train_nexamples, batches_per_epoch, nbatches))

        model.train()

        batch_num = 0
        while args.nepochs < 0 or batch_num < nbatches:
            batch_time_start = time.time()
            syms, feats_other, feats_audio, targets, target_weights = model.prepare_train_batch(charts_train, **feats_config)
            syms = torch.from_numpy(syms).to(DEVICE)
            feats_other = torch.from_numpy(feats_other).to(DEVICE)
            feats_audio = torch.from_numpy(feats_audio).to(DEVICE)
            targets = torch.from_numpy(targets).to(DEVICE)
            target_weights = torch.from_numpy(target_weights).to(DEVICE)
            targets = torch.flatten(targets)
            target_weights = torch.flatten(target_weights)
            # print(syms.size(), feats_other.size(), feats_audio.size(), targets.size(), target_weights.size())

            optimizer.zero_grad()
            output = model(x=syms, other=feats_other, audio=feats_audio)
            # print(output.size())
            loss = loss_function(output, targets)
            loss.backward()
            print('binary-xntrop-loss: ', end='')
            print(loss.item())
            optimizer.step()

            # epoch_xentropies.append(batch_xentropy)
            # epoch_times.append(time.time() - batch_time_start)

            batch_num += 1

            if batch_num % batches_per_epoch == 0:
                epoch_num = batch_num // batches_per_epoch
                print('Completed epoch {}'.format(epoch_num))

                # lr_decay = args.lr_decay_rate ** max(epoch_num - args.lr_decay_delay, 0)
                # lr_summary = model_train.assign_lr(sess, args.lr * lr_decay)
                # summary_writer.add_summary(lr_summary, batch_num)

                # epoch_xentropy = np.mean(epoch_xentropies)
                # print('Epoch mean cross-entropy (nats) {}'.format(epoch_xentropy))
                # epoch_summary = sess.run(epoch_summaries, feed_dict={epoch_mean_xentropy: epoch_xentropy, epoch_mean_time: np.mean(epoch_times), epoch_var_xentropy: np.var(epoch_xentropies), epoch_var_time: np.var(epoch_times), epoch_time_total: np.sum(epoch_times)})
                # summary_writer.add_summary(epoch_summary, batch_num)

                # epoch_xentropies = []
                # epoch_times = []

    # with tf.Graph().as_default(), tf.Session(config=config) as sess:
    #     # akiba csv
    #     csv = open(os.path.join(args.experiment_dir, 'train.csv'), mode='w')

    #     if do_train:
    #         print('Creating train model')
    #         with tf.variable_scope('model_ss', reuse=None):
    #             model_train = SymNet(mode='train', batch_size=args.batch_size, **model_config)

    #     if do_train_eval or do_eval:
    #         print('Creating eval model')
    #         with tf.variable_scope('model_ss', reuse=do_train):
    #             eval_batch_size = args.batch_size
    #             if args.rnn_size > 0 and args.rnn_nlayers > 0:
    #                 eval_batch_size = 1
    #             model_eval = SymNet(mode='eval', batch_size=eval_batch_size, **model_config)
    #             model_early_stop_xentropy_avg = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    #             model_early_stop_accuracy = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    #     if do_generate:
    #         print('Creating generation model')
    #         with tf.variable_scope('model_ss', reuse=do_train):
    #             eval_batch_size = args.batch_size
    #             model_gen = SymNet(mode='gen', batch_size=1, **model_config)

    #     # Restore or init model
    #     model_saver = tf.train.Saver(tf.global_variables())
    #     if args.model_ckpt_fp:
    #         print('Restoring model weights from {}'.format(args.model_ckpt_fp))
    #         model_saver.restore(sess, args.model_ckpt_fp)
    #     else:
    #         print('Initializing model weights from scratch')
    #         sess.run(tf.global_variables_initializer())

    #         # Restore or init sym weights
    #         if args.sym_rnn_pretrain_model_ckpt_fp:
    #             print('Restoring pretrained weights from {}'.format(args.sym_rnn_pretrain_model_ckpt_fp))
    #             var_list_old = list(filter(lambda x: 'nosym' not in x.name and 'cnn' not in x.name, tf.global_variables()))
    #             pretrain_saver = tf.train.Saver(var_list_old)
    #             pretrain_saver.restore(sess, args.sym_rnn_pretrain_model_ckpt_fp)

    #     # Create summaries
    #     if do_train:
    #         summary_writer = tf.summary.FileWriter(args.experiment_dir, sess.graph)

    #         epoch_mean_xentropy = tf.placeholder(tf.float32, shape=[], name='epoch_mean_xentropy')
    #         epoch_mean_time = tf.placeholder(tf.float32, shape=[], name='epoch_mean_time')
    #         epoch_var_xentropy = tf.placeholder(tf.float32, shape=[], name='epoch_var_xentropy')
    #         epoch_var_time = tf.placeholder(tf.float32, shape=[], name='epoch_var_time')
    #         epoch_time_total = tf.placeholder(tf.float32, shape=[], name='epoch_time_total')
    #         epoch_summaries = tf.summary.merge([
    #             tf.summary.scalar('epoch_mean_xentropy', epoch_mean_xentropy),
    #             tf.summary.scalar('epoch_mean_time', epoch_mean_time),
    #             tf.summary.scalar('epoch_var_xentropy', epoch_var_xentropy),
    #             tf.summary.scalar('epoch_var_time', epoch_var_time),
    #             tf.summary.scalar('epoch_time_total', epoch_time_total)
    #         ])

    #         eval_metric_names = ['xentropy_avg', 'accuracy']
    #         eval_metrics = {}
    #         eval_summaries = []
    #         for eval_metric_name in eval_metric_names:
    #             name_mean = 'eval_mean_{}'.format(eval_metric_name)
    #             name_var = 'eval_var_{}'.format(eval_metric_name)
    #             ph_mean = tf.placeholder(tf.float32, shape=[], name=name_mean)
    #             ph_var = tf.placeholder(tf.float32, shape=[], name=name_var)
    #             summary_mean = tf.summary.scalar(name_mean, ph_mean)
    #             summary_var = tf.summary.scalar(name_var, ph_var)
    #             eval_summaries.append(tf.summary.merge([summary_mean, summary_var]))
    #             eval_metrics[eval_metric_name] = (ph_mean, ph_var)
    #         eval_time = tf.placeholder(tf.float32, shape=[], name='eval_time')
    #         eval_time_summary = tf.summary.scalar('eval_time', eval_time)
    #         eval_summaries = tf.summary.merge([eval_time_summary] + eval_summaries)

    #         # Calculate epoch stuff
    #         train_nexamples = sum([chart.get_nannotations() for chart in charts_train])
    #         examples_per_batch = args.batch_size
    #         examples_per_batch *= model_train.out_nunroll
    #         batches_per_epoch = train_nexamples // examples_per_batch
    #         nbatches = args.nepochs * batches_per_epoch
    #         print('{} frames in data, {} batches per epoch, {} batches total'.format(train_nexamples, batches_per_epoch, nbatches))

    #         # Init epoch
    #         lr_summary = model_train.assign_lr(sess, args.lr)
    #         summary_writer.add_summary(lr_summary, 0)
    #         epoch_xentropies = []
    #         epoch_times = []

    #         batch_num = 0
    #         eval_best_xentropy_avg = float('inf')
    #         eval_best_accuracy = float('-inf')
    #         while args.nepochs < 0 or batch_num < nbatches:
    #             batch_time_start = time.time()
    #             syms, feats_other, feats_audio, targets, target_weights = model_train.prepare_train_batch(charts_train, **feats_config)
    #             feed_dict = {
    #                 model_train.syms: syms,
    #                 model_train.feats_other: feats_other,
    #                 model_train.feats_audio: feats_audio,
    #                 model_train.targets: targets,
    #                 model_train.target_weights: target_weights
    #             }
    #             batch_xentropy, _ = sess.run([model_train.avg_neg_log_lhood, model_train.train_op], feed_dict=feed_dict)

    #             epoch_xentropies.append(batch_xentropy)
    #             epoch_times.append(time.time() - batch_time_start)

    #             batch_num += 1

    #             if batch_num % batches_per_epoch == 0:
    #                 epoch_num = batch_num // batches_per_epoch
    #                 print('Completed epoch {}'.format(epoch_num))

    #                 lr_decay = args.lr_decay_rate ** max(epoch_num - args.lr_decay_delay, 0)
    #                 lr_summary = model_train.assign_lr(sess, args.lr * lr_decay)
    #                 summary_writer.add_summary(lr_summary, batch_num)

    #                 epoch_xentropy = np.mean(epoch_xentropies)
    #                 print('Epoch mean cross-entropy (nats) {}'.format(epoch_xentropy))
    #                 epoch_summary = sess.run(epoch_summaries, feed_dict={epoch_mean_xentropy: epoch_xentropy, epoch_mean_time: np.mean(epoch_times), epoch_var_xentropy: np.var(epoch_xentropies), epoch_var_time: np.var(epoch_times), epoch_time_total: np.sum(epoch_times)})
    #                 summary_writer.add_summary(epoch_summary, batch_num)

    #                 epoch_xentropies = []
    #                 epoch_times = []

    #             if batch_num % args.nbatches_per_ckpt == 0:
    #                 print('Saving model weights...')
    #                 ckpt_fp = os.path.join(args.experiment_dir, 'onset_net_train')
    #                 model_saver.save(sess, ckpt_fp, global_step=tf.contrib.framework.get_or_create_global_step())
    #                 print('Done saving!')

    #             if do_train_eval and batch_num % args.nbatches_per_eval == 0:
    #                 print('Evaluating...')
    #                 eval_start_time = time.time()

    #                 metrics = defaultdict(list)

    #                 for eval_chart in charts_valid:
    #                     if model_eval.do_rnn:
    #                         state = sess.run(model_eval.initial_state)

    #                     neg_log_prob_sum = 0.0
    #                     correct_predictions_sum = 0.0
    #                     weight_sum = 0.0
    #                     for syms, syms_in, feats_other, feats_audio, targets, target_weights in model_eval.eval_iter(eval_chart, **feats_config_eval):
    #                         feed_dict = {
    #                             model_eval.syms: syms_in,
    #                             model_eval.feats_other: feats_other,
    #                             model_eval.feats_audio: feats_audio,
    #                             model_eval.targets: targets,
    #                             model_eval.target_weights: target_weights
    #                         }
    #                         if model_eval.do_rnn:
    #                             feed_dict[model_eval.initial_state] = state
    #                             xentropies, correct_predictions, state = sess.run([model_eval.neg_log_lhoods, model_eval.correct_predictions, model_eval.final_state], feed_dict=feed_dict)
    #                         else:
    #                             xentropies, correct_predictions = sess.run([model_eval.neg_log_lhoods, model_eval.correct_predictions], feed_dict=feed_dict)

    #                         neg_log_prob_sum += np.sum(xentropies)
    #                         correct_predictions_sum += np.sum(correct_predictions)
    #                         weight_sum += np.sum(target_weights)

    #                     assert int(weight_sum) == eval_chart.get_nannotations()
    #                     xentropy_avg = neg_log_prob_sum / weight_sum
    #                     accuracy = correct_predictions_sum / weight_sum

    #                     metrics['xentropy_avg'].append(xentropy_avg)
    #                     metrics['accuracy'].append(accuracy)

    #                 metrics = {k: (np.mean(v), np.var(v)) for k, v in metrics.items()}
    #                 feed_dict = {}
    #                 results = []
    #                 for metric_name, (mean, var) in metrics.items():
    #                     feed_dict[eval_metrics[metric_name][0]] = mean
    #                     feed_dict[eval_metrics[metric_name][1]] = var
    #                 feed_dict[eval_time] = time.time() - eval_start_time

    #                 summary_writer.add_summary(sess.run(eval_summaries, feed_dict=feed_dict), batch_num)

    #                 xentropy_avg_mean = metrics['xentropy_avg'][0]
    #                 if xentropy_avg_mean < eval_best_xentropy_avg:
    #                     print('Xentropy {} better than previous {}'.format(xentropy_avg_mean, eval_best_xentropy_avg))
    #                     ckpt_fp = os.path.join(args.experiment_dir, 'onset_net_early_stop_xentropy_avg')
    #                     model_early_stop_xentropy_avg.save(sess, ckpt_fp, global_step=tf.contrib.framework.get_or_create_global_step())
    #                     eval_best_xentropy_avg = xentropy_avg_mean

    #                 accuracy_mean = metrics['accuracy'][0]
    #                 if accuracy_mean > eval_best_accuracy:
    #                     print('Accuracy {} better than previous {}'.format(accuracy_mean, eval_best_accuracy))
    #                     ckpt_fp = os.path.join(args.experiment_dir, 'onset_net_early_stop_accuracy')
    #                     model_early_stop_accuracy.save(sess, ckpt_fp, global_step=tf.contrib.framework.get_or_create_global_step())
    #                     eval_best_accuracy = accuracy_mean

    #                 csv.write(','.join([str(xentropy_avg_mean),str(accuracy_mean)]) + '\n')

    #                 print('Done evaluating')

    #     if do_eval:
    #         print('Evaluating...')

    #         metrics = defaultdict(list)

    #         for test_chart in charts_test:
    #             if model_eval.do_rnn:
    #                 state = sess.run(model_eval.initial_state)

    #             neg_log_prob_sum = 0.0
    #             correct_predictions_sum = 0.0
    #             weight_sum = 0.0
    #             for syms, syms_in, feats_other, feats_audio, targets, target_weights in model_eval.eval_iter(test_chart, **feats_config_eval):
    #                 feed_dict = {
    #                     model_eval.syms: syms_in,
    #                     model_eval.feats_other: feats_other,
    #                     model_eval.feats_audio: feats_audio,
    #                     model_eval.targets: targets,
    #                     model_eval.target_weights: target_weights
    #                 }
    #                 if model_eval.do_rnn:
    #                     feed_dict[model_eval.initial_state] = state
    #                     xentropies, correct_predictions, state = sess.run([model_eval.neg_log_lhoods, model_eval.correct_predictions, model_eval.final_state], feed_dict=feed_dict)
    #                 else:
    #                     xentropies, correct_predictions = sess.run([model_eval.neg_log_lhoods, model_eval.correct_predictions], feed_dict=feed_dict)

    #                 neg_log_prob_sum += np.sum(xentropies)
    #                 correct_predictions_sum += np.sum(correct_predictions)
    #                 weight_sum += np.sum(target_weights)

    #             assert int(weight_sum) == test_chart.get_nannotations()
    #             xentropy_avg = neg_log_prob_sum / weight_sum
    #             accuracy = correct_predictions_sum / weight_sum

    #             metrics['perplexity'].append(np.exp(xentropy_avg))
    #             metrics['xentropy_avg'].append(xentropy_avg)
    #             metrics['accuracy'].append(accuracy)

    #         metrics = {k: (np.mean(v), np.std(v), np.min(v), np.max(v)) for k, v in metrics.items()}
    #         copy_pasta = []
    #         for metric_name in ['xentropy_avg', 'perplexity', 'accuracy']:
    #             metric_stats = metrics[metric_name]
    #             copy_pasta += list(metric_stats)
    #             print('{}: {}'.format(metric_name, metric_stats))
    #         print('COPY PASTA:')
    #         print(','.join([str(x) for x in copy_pasta]))

    #     # TODO: This currently only works for VERY specific model (delta time LSTM)
    #     if do_generate:
    #         print('Generating...')

    #         with open(args.generate_fp, 'r') as f:
    #             step_times = [float(x) for x in f.read().split(',')]

    #         with open(args.generate_vocab_fp, 'r') as f:
    #             idx_to_sym = {i:k for i, k in enumerate(f.read().splitlines())}

    #         def weighted_pick(weights):
    #             t = np.cumsum(weights)
    #             s = np.sum(weights)
    #             return(int(np.searchsorted(t, np.random.rand(1)*s)))

    #         state = sess.run(model_gen.initial_state)
    #         sym_prev = '<-1>'
    #         step_time_prev = step_times[0]
    #         seq_scores = []
    #         seq_sym_idxs = []
    #         seq_syms = []
    #         for step_time in step_times:
    #             delta_step_time = step_time - step_time_prev

    #             syms_in = np.array([[model_gen.arrow_to_encoding(sym_prev, 'bagofarrows')]], dtype=np.float32)
    #             feats_other = np.array([[[delta_step_time]]], dtype=np.float32)
    #             feats_audio = np.zeros((1, 1, 0, 0, 0), dtype=np.float32)
    #             feed_dict = {
    #                 model_gen.syms: syms_in,
    #                 model_gen.feats_other: feats_other,
    #                 model_gen.feats_audio: feats_audio,
    #                 model_gen.initial_state: state
    #             }

    #             scores, state = sess.run([model_gen.scores, model_gen.final_state], feed_dict=feed_dict)

    #             sym_idx = 0
    #             while sym_idx <= 1:
    #                 sym_idx = weighted_pick(scores)
    #                 if sym_idx <= 1:
    #                     print('rare')
    #             sym_idx = sym_idx - 1 # remove special
    #             sym = idx_to_sym[sym_idx]

    #             seq_scores.append(scores)
    #             seq_sym_idxs.append(sym_idx)
    #             seq_syms.append(sym)

    #             sym_prev = sym
    #             step_time_prev = step_time

    #         with open(os.path.join(args.experiment_dir, 'seq.pkl'), 'wb') as f:
    #             pickle.dump((seq_scores, seq_sym_idxs, seq_syms), f)

if __name__ == '__main__':
    main()
