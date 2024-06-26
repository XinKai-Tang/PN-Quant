def get_anchors(bases=[5, 10, 20, 30, 50],
                aspect_ratios=[[1, 1, 1]]):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])
    return anchors


net_cfgs = {
    # Net configuration
    'anchors': get_anchors(),
    'chanel': 1,
    'crop_size': [96, 96, 96],
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,
    'th_pos_train': 0.5,
    'th_pos_val': 1,
    'num_hard': 3,
    'bound_size': 12,
    'blacklist': [],

    'augtype': {'flip': True, 'rotate': True, 'scale': True, 'swap': False},
    'r_rand_crop': 0.,
    'pad_value': 170,

    ## region proposal network configuration
    'rpn_train_bg_thresh_high': 0.02,
    'rpn_train_fg_thresh_low': 0.5,

    'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.1,
    'rpn_test_nms_pre_score_threshold': 0.5,
    'rpn_test_nms_overlap_threshold': 0.1,

    # false positive reduction network configuration
    'num_class': 2,
    'rcnn_crop_size': (7, 7, 7),  # can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.1,
    'rcnn_train_batch_size': 64,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.1,
    'rcnn_test_nms_pre_score_threshold': 0.0,
    'rcnn_test_nms_overlap_threshold': 0.1,

    'mask_crop_size': [24, 48, 48],
    'mask_test_nms_overlap_threshold': 0.3,

    'box_reg_weight': [1., 1., 1., 1., 1., 1.]
}
