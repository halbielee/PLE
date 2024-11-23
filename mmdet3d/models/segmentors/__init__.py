# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DSegmentor
from .cylinder3d import Cylinder3D
from .encoder_decoder import EncoderDecoder3D
from .lasermix import LaserMix
from .minkunet import MinkUNet
from .seg3d_tta import Seg3DTTAModel
from .semi_base import SemiBase3DSegmentor

from .mean_teacher import MeanTeacher3DSegmentor
from .semi_mono import Semi3DSegmentor

# newly added
from .semi_dual import SemiDualBranch3DSegmentor
from .mean_teacher_dualbranch import MeanTeacherDualBranch3DSegmentor
from .lasermix_dualbranch import LaserMixDualBranch


__all__ = [
    'Base3DSegmentor', 'EncoderDecoder3D', 'Cylinder3D', 'MinkUNet',
    'Seg3DTTAModel', 'SemiBase3DSegmentor', 'LaserMix',
    'Semi3DSegmentor', 'MeanTeacher3DSegmentor', 
    
    # newly added
    # Base model + DualBranch
    'SemiDualBranch3DSegmentor', 
    # Mean teacher + DualBranch
    'MeanTeacherDualBranch3DSegmentor',
    # LaserMix + DualBranch
    'LaserMixDualBranch',
]
