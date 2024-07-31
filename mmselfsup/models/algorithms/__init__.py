# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseModel
from .byol import BYOL
from .classification import Classification
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .moco import MoCo
from .mocov3 import MoCoV3
from .npid import NPID
from .odc import ODC
from .relative_loc import RelativeLoc
from .rotation_pred import RotationPred
from .simclr import SimCLR
from .simsiam import SimSiam
from .swav import SwAV
from .mae import MAE
from .my_montage_rec_v22 import MY_MoCo_V22
from .my_montage_rec_v3 import MY_MoCo_V3
from .my_montage_rec_v4 import MY_MoCo_v4
from .my_montage_rec_multi_v5 import MY_MoCo_v5
from .my_montage_rec_v6 import MY_MOCO_V6
from .my_montage_rec_v62 import MY_MOCO_V62
from .my_montage_rec_v64 import MY_MOCO_V64
from .my_montage_rec_v7 import MY_MOCO_V7
from .my_global_loc_v0 import MY_MoCo_V90
from.my_global_loc_v01 import MY_MoCo_V91
from .my_global_loc_rec_v1 import MY_MoCo_V9
from .my_global_loc_rec_v2 import MY_MoCo_V92
from .my_global_loc_rec_v3 import MY_MoCo_V93
from .my_global_loc_rec_v4 import MY_MoCo_V94
from .my_global_loc_rec_v5 import MY_MoCo_V95
from .my_global_loc_rec_v6 import MY_MoCo_V96
from .my_global_loc_rec_v7 import MY_MoCo_V11

from .simmim import SimMIM
from .corev2 import CoRev2

__all__ = [
    'BaseModel', 'BYOL', 'Classification', 'DeepCluster', 'DenseCL', 'MoCo',
    'MoCoV3', 'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR',
    'SimSiam', 'SwAV','MY_MoCo_V3','MY_MoCo_v4','MY_MoCo_v5','MY_MoCo_V22','MY_MOCO_V6','MY_MOCO_V7','MY_MOCO_V62','MY_MOCO_V64',
    'MY_MoCo_V90','MY_MoCo_V91','MY_MoCo_V9','MY_MoCo_V92','MY_MoCo_V93','MY_MoCo_V94','MY_MoCo_V95','MY_MoCo_V96','MY_MoCo_V11','MAE','SimMIM',
    'CoRev2'
]
