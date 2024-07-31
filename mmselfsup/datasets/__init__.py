# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDataset
from .builder import (DATASETS, DATASOURCES, PIPELINES, build_dataloader,
                      build_dataset, build_datasource)
from .data_sources import *  # noqa: F401,F403
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .deepcluster import DeepClusterDataset
from .multi_view import MultiViewDataset
from .my_view import MontageViewDataset   ## center transform
# from .my_view_subfig import MontageViewDataset   ### subfig cutout
# from .my_view_cutcenter import MontageViewDataset
from .Montage_rec_loc import MontageLocViewDataset
from .pipelines import *  # noqa: F401,F403
from .relative_loc import RelativeLocDataset
from .rotation_pred import RotationPredDataset
from .samplers import *  # noqa: F401,F403
from .single_view import SingleViewDataset
from .my_view_trancenter import MontageViewPILDataset
from .my_view_randomchose_and_tc import RandomchoiceAndTCView
from .my_view_randomchose_and_tc_v2 import RandomchoiceAndRandcutView
from .my_view_randomchose_and_tc_v0 import RandomchoiceAndRandcutViewV0
from .my_view_randomchose_and_tc_v5 import RandomchoiceAndTCViewV5
from .my_view_randomchose_and_tc_v7 import RandomchoiceAndTCViewV7  ### 适用于 v11

from .three_view import ThreeViewDataset ### 适用于core-v2


__all__ = [
    'DATASETS', 'DATASOURCES', 'PIPELINES', 'BaseDataset', 'build_dataloader',
    'build_dataset', 'build_datasource', 'ConcatDataset', 'RepeatDataset',
    'DeepClusterDataset', 'MultiViewDataset', 'SingleViewDataset',
    'RelativeLocDataset', 'RotationPredDataset','MontageViewDataset','MontageViewPILDataset','RandomchoiceAndTCView',
    'RandomchoiceAndRandcutView','RandomchoiceAndRandcutViewV0','RandomchoiceAndTCViewV5','RandomchoiceAndTCViewV7',
    'ThreeViewDataset',
]
