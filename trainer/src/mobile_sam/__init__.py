# Vendored from MobileSAM (https://github.com/ChaoningZhang/MobileSAM)
# Licensed under the Apache License, Version 2.0 (see LICENSE_MOBILE_SAM)
#
# Modified: removed unused components (ViT-H/L/B, predictor,
# automatic mask generator), replaced timm dependency with inline equivalents.

from .build_sam import build_sam_vit_t
