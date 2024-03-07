# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Shahaf Arica from https://github.com/facebookresearch/CutLER/blob/main/cutler/config/cutler_config.py

def add_cuvler_config(cfg):
    cfg.DATALOADER.COPY_PASTE = False
    cfg.DATALOADER.COPY_PASTE_RATE = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE = False
    cfg.DATALOADER.COPY_PASTE_PROPORTIONS = (0.2, 0.4)

    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
    cfg.MODEL.ROI_HEADS.COSINE_SCALE = 20.0
    cfg.MODEL.BACKBONE.FREEZE = False
    cfg.MODEL.ROI_HEADS.FREEZE_FEAT = False
    cfg.MODEL.PROPOSAL_GENERATOR.FREEZE = False

    cfg.MODEL.ROI_HEADS.USE_SOFT_TARGETS = False

    cfg.SOLVER.BASE_LR_MULTIPLIER = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES = []

    """
    Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.
    """

    cfg.TEST.NO_SEGM = False
