#!/bin/sh

for yaml in \
PURE_UNSUPERVISED.yaml \
UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml \
UBFC-rPPG_PURE_EFFICIENTPHYS.yaml \
UBFC-rPPG_PURE_FactorizePhys_FSAM_Res.yaml \
UBFC-rPPG_PURE_PHYSFORMER_BASIC.yaml \
UBFC-rPPG_PURE_PHYSMAMBA_BASIC.yaml \
UBFC-rPPG_PURE_PHYSNET_BASIC.yaml \
UBFC-rPPG_PURE_RHYTHMFORMER.yaml \
UBFC-rPPG_PURE_TSCAN_BASIC.yaml
do
    python main.py --config_file "configs/infer_configs/$yaml"
done
