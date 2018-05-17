#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module

import os
import numpy as np
import tensorflow as tf
import texar as tx

from texar.modules.encoders.hierarchical_encoders_new import HierarchicalRNNEncoder
from texar.modules.decoders.beam_search_decode import beam_search_decode

from tensorflow.contrib.seq2seq import tile_batch

from argparse import ArgumentParser

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sw_loader import download_and_process

parser = ArgumentParser()
parser.add_argument('-l', '--load_path', default=None, type=str)
parser.add_argument('--stage', nargs='+',
                    default=['train', 'val', 'test'], type=str)
parser.add_argument('--test_batch_num', default=None, type=int)
parser.add_argument('--data_root', default='../../data/sw1c2r/', type=str)
parser.add_argument('--save_root', default='./save', type=str)
args = parser.parse_args()

download_and_process(args.data_root)

UTTRCNT = 10

data_hparams = {
    stage: {
        "num_epochs": 1,
        "shuffle": stage != 'test',
        "batch_size": 30,
        "datasets": [
            { # source
                "variable_utterance": True,
                "max_utterance_cnt": UTTRCNT - 1,
                "files": [
                    os.path.join(args.data_root, '{}-source.txt'.format(stage))],
                "vocab_file": os.path.join(args.data_root, 'vocab.txt'),
                "embedding_init": {
                    "file": os.path.join(args.data_root, 'embedding.txt'),
                    "dim": 200,
                    "read_fn": "load_glove"
                },
                "data_name": "source"
            },
            { # target
                "files": [
                    os.path.join(args.data_root, '{}-target.txt'.format(stage))],
                "vocab_share_with": 0,
                "data_name": "target"
            },
        ] + [{
                "files": os.path.join(args.data_root, '{}-source-spk-{}.txt'.format(stage, i)),
                "data_type": "float", # int actually
                "data_name": "spk_{}".format(i)
            } for i in range(UTTRCNT - 1)
        ] + [{
                "files": os.path.join(args.data_root, '{}-target-spk.txt'.format(stage)),
                "data_type": "float", # int actually
                "data_name": "spk_tgt"
            }
        ] + [{
                "variable_utterance": True,
                "max_utterance_cnt": 10,
                "files": [os.path.join(args.data_root, '{}-target-refs.txt'.format(stage))],
                "vocab_share_with": 0,
                "data_name": "refs"
            }]
    }
    for stage in ['train', 'val', 'test']
}

encoder_minor_hparams = {
    "rnn_cell_fw": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": 300,
            "kernel_initializer": tf.orthogonal_initializer(),
        },
        "dropout": {
            "input_keep_prob": 1,
        }
    },
    "rnn_cell_share_config": True
}
encoder_major_hparams = {
    "rnn_cell": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": 600,
            "kernel_initializer": tf.orthogonal_initializer(),
        },
    },
}
decoder_hparams = {
    "num_units": 200, 
    "poswise_feedforward": {
        "name": 'ffn',
        "layers": [
            {
                "type": "Dense",
                "kwargs": {
                    "name": "conv1",
                    "units": 200,
                    "activation": 'relu',
                    "use_bias": True,
                }
            },
            {
                "type": "Dropout",
                "kwargs": {
                    "rate": 0.3,
                }
            },
            {
                "type": "Dense",
                "kwargs": {
                    "name": "conv2",
                    "units": 200,
                    "use_bias": True,
                }
            }
        ],
    },
    "num_blocks": 6, 
    "beam_width": 1, 
    "sampling_method": 'sample',
    "maximum_decode_length": 50
}
opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001,
        }
    }
}

def main():
    train_data = tx.data.MultiAlignedData(data_hparams['train'])
    val_data = tx.data.MultiAlignedData(data_hparams['val'])
    test_data = tx.data.MultiAlignedData(data_hparams['test'])
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()
    spk_src = tf.stack([data_batch['spk_{}'.format(i)] for i in range(UTTRCNT-1)], 1)
    spk_tgt = data_batch['spk_tgt']

    # declare modules

    embedder = tx.modules.WordEmbedder(
        init_value=train_data.embedding_init_value(0).word_vecs)

    encoder_minor = tx.modules.BidirectionalRNNEncoder(
        hparams=encoder_minor_hparams)
    encoder_major = tx.modules.UnidirectionalRNNEncoder(
        hparams=encoder_major_hparams)
    encoder = HierarchicalRNNEncoder(
        encoder_major, encoder_minor)
    encoder_medium = lambda x: tf.concat([x, tf.reshape(spk_src, (-1, 1))], 1)

    decoder = tx.modules.TransformerDecoder(
        embedding=embedder.embedding, 
        hparams=decoder_hparams, vocab_size=train_data.vocab(0).size)

    #connector = tf.layers.Dense(decoder.cell.state_size) 

    connector = tx.modules.connectors.MLPTransformConnector(
        decoder.hparams.num_units)

    # build graph

    dialog_embed = embedder(data_batch['source_text_ids'])

    ecdr_outputs, ecdr_states = encoder(
        dialog_embed,
        medium_after_depack=encoder_medium,
        sequence_length=data_batch['source_length'],
        sequence_length_major=data_batch['source_utterance_cnt'])[:2]

    #ecdr_outputs = tf.concat(ecdr_outputs, axis=-1)

    ecdr_states = (ecdr_states, ) + (tf.reshape(spk_tgt, (-1, 1)), )

    dcdr_states = connector(ecdr_states)

    ecdr_outputs = tf.layers.dense(
        ecdr_outputs, 200, tf.nn.tanh, use_bias=True)
    
    from texar.core import attentions
    uttr_padding = 1. - tf.to_float(tf.sequence_mask(
        data_batch['source_utterance_cnt']))
    ignore_padding = attentions.attention_bias_ignore_padding(uttr_padding)
    encoder_decoder_attention_bias = ignore_padding

    # train branch
    outputs, _ = decoder(
        decoder_input=data_batch['target_text_ids'],
        encoder_output=ecdr_outputs,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias)

    # call decoder.trainable_variables

    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['target_text_ids'][:, 1:],
        logits=outputs[:, :-1],
        sequence_length=data_batch['target_length'] - 1,
        sum_over_timesteps=False,
        average_across_timesteps=True)

    global_step = tf.Variable(0, name='global_step', trainable=True)
    train_op = tx.core.get_train_op(
        mle_loss, global_step=global_step, hparams=opt_hparams)

    # test branch

    test_batch_size = test_data.hparams.batch_size

    # sample inference

    #dcdr_states_tiled = tile_batch(dcdr_states, 5)
    output_samples = [decoder.dynamic_decode(
        ecdr_outputs, encoder_decoder_attention_bias)['sampled_ids']
        for i in range(5)]

    # denumericalize the generated samples
    sample_text = [train_data.vocab(0).map_ids_to_tokens(
        output_sample) for output_sample in output_samples]

    target_tuple = (data_batch['target_text'][:, 1:],
                    data_batch['target_length'] - 1,
                    data_batch['target_text_ids'][:, 1:])
    #train_data.source_vocab.map_ids_to_tokens(
    #data_batch['target_text_ids'][:, 1:]),
    #data_batch['target_length'] - 1)

    dialog_tuple = (data_batch['source_text'], data_batch['source_length'],
                    data_batch['source_utterance_cnt'])

    refs_tuple = (data_batch['refs_text'][:, :, 1:], data_batch['refs_length'],
                  data_batch['refs_text_ids'][:, :, 1:], data_batch['refs_utterance_cnt'])

    def _train_epochs(sess, epoch, display=10):
        iterator.switch_to_train_data(sess)

        for i in range(3000): # speed up a epoch.
        #while True:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                step, loss, _ = sess.run(
                    [global_step, mle_loss, train_op], feed_dict=feed)

                if step % display == 0:
                    print('step {} at epoch {}: loss={}'.format(
                        step, epoch, loss))

            except tf.errors.OutOfRangeError:
                break

        print('epoch {} train fin: loss={}'.format(epoch, loss))

    def _val_epochs(sess, epoch, loss_histories):
        iterator.switch_to_val_data(sess)

        valid_loss = []
        cnt = 0
        while True:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}
                loss = sess.run(mle_loss, feed_dict=feed)
                valid_loss.append(loss)

            except tf.errors.OutOfRangeError:
                loss = np.mean(valid_loss)
                print('epoch {} valid fin: loss={}'.format(epoch, loss))
                break

        loss_histories.append(loss)
        best = min(loss_histories)

        return len(loss_histories) - loss_histories.index(best) - 1

    def _test_epochs(sess, epoch, test_batch_num=None):
        iterator.switch_to_test_data(sess)

        max_bleus = [[] for i in range(5)]
        avg_bleus = [[] for i in range(5)]
        max_As = []
        avg_As = []
        max_Es = []
        avg_Es = []
        txt_results = []

        batch_cnt = 0

        from scipy import spatial

        def Abow(hyps, refs, embedding):
            a = [sum(embedding[i] for i in ref) / len(ref) for ref in refs]
            b = [sum(embedding[i] for i in hyp) / len(hyp) for hyp in hyps]

            prec = np.mean([max([spatial.distance.cosine(x, y) for x in a]) for y in b])
            recall = np.mean([max([spatial.distance.cosine(x, y) for y in b]) for x in a])

            return prec, recall

        def Ebow(hyps, refs, embedding):
            def extrema(x):
                r = []
                for i in range(200):
                    a = min([t[i] for t in x])
                    b = max([t[i] for t in x])
                    if abs(a) > b:
                        r.append(a)
                    else:
                        r.append(b)
                return np.array(r)

            a = [extrema([embedding[i] for i in ref]) / len(ref) for ref in refs]
            b = [extrema([embedding[i] for i in hyp]) / len(hyp) for hyp in hyps]

            prec = np.mean([max([spatial.distance.cosine(x, y) for x in a]) for y in b])
            recall = np.mean([max([spatial.distance.cosine(x, y) for y in b]) for x in a])

            return prec, recall

        def BLEU(hyps, refs, weight):
            prec = np.mean([max([sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7, weights=weights) for ref in refs]) for hyp in hyps])
            recall = np.mean([max([sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7, weights=weights) for hyp in hyps]) for ref in refs])

            return prec, recall

        while batch_cnt != test_batch_num:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}

                #samples = sess.run(sample_text, feed_dict=feed)

                output, samples, sample_id, dialog_t, target_t, refs_t = sess.run(
                    [outputs, sample_text, output_samples,
                     dialog_tuple, target_tuple, refs_tuple],
                    feed_dict=feed)

                max_length = max(x.shape[2] for x in samples)
                samples = np.concatenate([np.pad(
                    x, ((0, 0), (0, 0), (0, max_length - x.shape[2])), 'constant')
                    for x in samples], axis=1)

                sample_id = np.concatenate([np.pad(
                    x, ((0, 0), (0, 0), (0, max_length - x.shape[2])), 'constant')
                    for x in sample_id], axis=1)

                #if sample inference used
                samples = samples.transpose(0, 2, 1)
                sample_id = sample_id.transpose(0, 2, 1)

                lengths = np.array(
                    [[(sample_id[j, :, i].tolist() + [2]).index(2) + 1 for j in range(sample_id.shape[0])] for i in range(5)]).transpose()

                for (beam, beam_len, beam_ids,
                     dialog, utts_len, utts_cnt,
                     target, tgt_len, tgt_ids,
                     refs, refs_len, refs_ids, refs_cnt) in zip(
                    samples, lengths, sample_id, *dialog_t, *target_t, *refs_t):

                    srcs = [dialog[i, :utts_len[i]] for i in range(utts_cnt)]
                    hyps = [beam[:l-1, i] for i, l in enumerate(beam_len)]
                    hyps_ids = [beam_ids[:l-1, i] for i, l in enumerate(beam_len)]
                    refs = [refs[i, :refs_len[i]-1] for i in range(refs_cnt)][:6]
                    refs += [target[:tgt_len-1]]
                    refs_ids = [refs_ids[i, :refs_len[i]-1] for i, l in enumerate(range(refs_cnt))][:6]
                    refs_ids += [tgt_ids[:tgt_len-1]]

                    hyps_ids = [hyp_ids for hyp_ids in hyps_ids if len(hyp_ids) > 0]

                    embedding = test_data.embedding_init_value(0).word_vecs
                    avg_A, max_A = Abow(hyps_ids, refs_ids, embedding)
                    max_As.append(max_A)
                    avg_As.append(avg_A)
                    avg_E, max_E = Ebow(hyps_ids, refs_ids, embedding)
                    max_Es.append(max_E)
                    avg_Es.append(avg_E)

                    for bleu_i in range(1, 5):
                        weights = [1. / bleu_i, ] * bleu_i

                        scrs = []

                        for hyp in hyps:
                            try:    
                                scrs.append(sentence_bleu(refs, hyp,
                                    smoothing_function=SmoothingFunction().method7,
                                    weights=weights))
                            except:
                                pass
                                #scrs.append(0)

                        if len(scrs) == 0:
                            scrs.append(0)

                        max_bleu, avg_bleu = np.max(scrs), np.mean(scrs)
                        max_bleus[bleu_i].append(max_bleu)
                        avg_bleus[bleu_i].append(avg_bleu)

                    #weights = [1/4., 1/4., 1/4., 1/4.]

                    #max_bleu, avg_bleu = np.max(scrs), np.mean(scrs)

                    #max_bleus.append(max_bleu)
                    #vg_bleus.append(avg_bleu)

                    src_txt = b'\n'.join([b' '.join(s[1:-1]) for s in srcs])
                    hyp_txt = b'\n'.join([b' '.join(s) for s in hyps])
                    ref_txt = b'\n'.join([b' '.join(s) for s in refs])
                    txt_results.append('input:\n{}\nhyps:\n{}\nref:\n{}'.format(
                        src_txt.decode(), hyp_txt.decode(), ref_txt.decode()))

                for i in range(1, 5):
                    print(np.mean(avg_bleus[i]), np.mean(max_bleus[i]))

            except tf.errors.OutOfRangeError:
                break

            batch_cnt += 1
            print('test batch {}/{}'.format(
                batch_cnt, test_batch_num))

        As_recall = np.mean(max_As)
        As_prec = np.mean(avg_As)
        Es_recall = np.mean(max_Es)
        Es_prec = np.mean(avg_Es)
        bleu_recall = [np.mean(max_bleus[i]) for i in range(1, 5)]
        bleu_prec = [np.mean(avg_bleus[i]) for i in range(1, 5)]
        #bleu_recall = np.mean(max_bleus)
        #bleu_prec = np.mean(avg_bleus)

        print('epoch {} test fin: As_recall={}, As_pred={}, Es_recall={}, Es_prec={}'.format(
            epoch, As_recall, As_prec, Es_recall, Es_prec)) 

        for i in range(1, 5):
            print('BLEU-{} prec={}, BLEU-{} recall={}'.format(i, bleu_prec[i-1], i, bleu_recall[i-1])) 

        with open('test_txt_results.txt', 'w') as f:
            f.write('\n\n'.join(txt_results))


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if args.load_path:
            saver.restore(sess, args.load_path)

        loss_histories = []

        for epoch in range(100):
            if 'train' in args.stage:
                assert 'val' in args.stage
                _train_epochs(sess, epoch)
            if 'val' in args.stage:
                best_index_diff = _val_epochs(sess, epoch, loss_histories)
            if 'test' in args.stage:
                _test_epochs(sess, epoch, args.test_batch_num)

            if 'train' in args.stage:
                if best_index_diff == 0:
                    saver.save(sess, os.path.join(args.save_root, 'hierarchical_example_best.ckpt'))
                elif best_index_diff > 15:
                    print('overfit at epoch {}'.format(epoch))
                    break
            else:
                break

if __name__ == "__main__":
    main()
