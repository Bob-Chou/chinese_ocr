from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from ..roi_data_layer.layer import RoIDataLayer
from ..utils.timer import Timer
from ..roi_data_layer import roidb as rdl_roidb
from ..fast_rcnn.config import cfg
from ..networks.VGGnet_train import VGGnet_train
from zoo import init_nncontext
from zoo.tfpark import TFDataset
from zoo.tfpark.estimator import TFEstimator, TFEstimatorSpec

_DEBUG = False
sc = init_nncontext()

class SolverWrapper(object):
    def __init__(self, imdb, roidb, output_dir, logdir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print('Computing bounding-box regression targets...')
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print('done')

        # For checkpoint
        self.logdir =logdir
        # self.saver = tf.train.Saver(max_to_keep=100,write_version=tf.train.SaverDef.V2)
        # self.writer = tf.summary.FileWriter(logdir=logdir,
        #                                      graph=tf.get_default_graph(),
        #                                      flush_secs=5)


    # def snapshot(self, sess, iter):
    #     net = self.net
    #     if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
    #         # save original values
    #         with tf.variable_scope('bbox_pred', reuse=True):
    #             weights = tf.get_variable("weights")
    #             biases = tf.get_variable("biases")
    #
    #         orig_0 = weights.eval()
    #         orig_1 = biases.eval()
    #
    #         # scale and shift with bbox reg unnormalization; then save snapshot
    #         weights_shape = weights.get_shape().as_list()
    #         sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
    #         sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))
    #
    #     if not os.path.exists(self.output_dir):
    #         os.makedirs(self.output_dir)
    #
    #     infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
    #              if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
    #     filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
    #                 '_iter_{:d}'.format(iter+1) + '.ckpt')
    #     filename = os.path.join(self.output_dir, filename)
    #
    #     self.saver.save(sess, filename)
    #     print('Wrote snapshot to: {:s}'.format(filename))
    #
    #     if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers:
    #         # restore net to original state
    #         sess.run(weights.assign(orig_0))
    #         sess.run(biases.assign(orig_1))

    def build_image_summary(self):
        # A simple graph for write image summary

        log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
        log_image_name = tf.placeholder(tf.string)
        # import tensorflow.python.ops.gen_logging_ops as logging_ops
        from tensorflow.python.ops import gen_logging_ops
        from tensorflow.python.framework import ops as _ops
        log_image = gen_logging_ops.image_summary(log_image_name, tf.expand_dims(log_image_data, 0), max_images=1)
        _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
        # log_image = tf.summary.image(log_image_name, tf.expand_dims(log_image_data, 0), max_outputs=1)
        return log_image, log_image_data, log_image_name

    def train_model(self, max_iters, restore=False):
        """Network training loop."""

        def model_fn(features, labels, mode):
            net = VGGnet_train(features, labels, mode)
            return net.model_fn()

        def input_fn(mode):
            from ..roi_data_layer.minibatch import get_minibatch

            def map_data(blob):
                """Selected used features and strip the batchsize dim"""
                return blob['data'][0], \
                       {'im_info': blob['im_info'][0],
                        'gt_boxes': blob['gt_boxes'],
                        'gt_ishard': blob['gt_ishard'],
                        'dontcare_areas': blob['dontcare_areas']}
            # Retrieve all training data
            print('Retrieving all training data')
            data = [get_minibatch([roi], self.imdb.num_classes)
                    for roi in self.roidb[:100]]
            features, labels = zip(*list(map(map_data, data)))
            print('done')
            if mode == tf.estimator.ModeKeys.TRAIN:
                return TFDataset.from_rdd(
                    sc.parallelize(features).zip(sc.parallelize(labels)),
                    features=(tf.float32, [None, None, 3]),
                    labels={'im_info': (tf.float32, [3]),
                            'gt_boxes': (tf.float32, [None, 5]),
                            'gt_ishard': (tf.int32, [None]),
                            'dontcare_areas': (tf.float32, [None, 4])},
                    batch_size=28)
            else:
                raise NotImplementedError

        estimator = TFEstimator(model_fn, tf.train.AdamOptimizer(),
                                model_dir=self.output_dir)
        estimator.train(input_fn, steps=max_iters)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    if cfg.TRAIN.HAS_RPN:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            # obsolete
            # layer = GtDataLayer(roidb)
            raise "Calling caffe modules..."
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer



def train_net(imdb, roidb, output_dir, log_dir, pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN network."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(imdb, roidb, output_dir, logdir= log_dir, pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(max_iters, restore=restore)
        print('done solving')
