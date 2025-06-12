#!/bin/sh

for yaml in \
PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml \
PURE_UBFC-rPPG_EFFICIENTPHYS.yaml \
PURE_UBFC-rPPG_FactorizePhys_FSAM_Res.yaml \
PURE_UBFC-rPPG_iBVPNet_BASIC.yaml \
PURE_UBFC-rPPG_PHYSFORMER_BASIC.yaml \
PURE_UBFC-rPPG_PHYSMAMBA_BASIC.yaml \
PURE_UBFC-rPPG_PHYSNET_BASIC.yaml \
PURE_UBFC-rPPG_RHYTHMFORMER_BASIC.yaml \
PURE_UBFC-rPPG_TSCAN_BASIC.yaml \
UBFC-rPPG_UNSUPERVISED.yaml
do
    python main.py --config_file "configs/infer_configs/$yaml"
done
