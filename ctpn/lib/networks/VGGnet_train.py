#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from .network import Network
from ..fast_rcnn.config import cfg
from zoo.tfpark.estimator import TFEstimatorSpec

class VGGnet_train(Network):
    def __init__(self, features=None, labels=None, mode=None, trainable=True):
        self.inputs = []
        if features is None or labels is None or mode is None:
            self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
            self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
            self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
            self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
            self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
            self.keep_prob = tf.placeholder(tf.float32)
        else:
            self.mode = mode
            self.data = features
            self.im_info = labels['im_info']
            self.keep_prob = tf.constant(0.5)
            self.gt_boxes = labels['gt_boxes']
            self.gt_ishard = labels['gt_ishard']
            self.dontcare_areas = labels['dontcare_areas']
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes,\
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):

        # n_classes = 21
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3'))

        (self.feed('rpn_conv/3x3').Bilstm(512,128,512,name='lstm_o'))
        (self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 2,name='rpn_cls_score'))

        # generating training labels on the fly
        # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
        # 给每个anchor上标签，并计算真值（也是delta的形式），以及内部权重和外部权重
        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))

        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             .spatial_softmax(name='rpn_cls_prob'))

    def model_fn(self):
        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = self.build_loss(
            ohem=cfg.TRAIN.OHEM)
        if self.mode == tf.estimator.ModeKeys.EVAL or self.mode == tf.estimator.ModeKeys.TRAIN:
            return TFEstimatorSpec(self.mode, loss=total_loss)
        else:
            raise NotImplementedError