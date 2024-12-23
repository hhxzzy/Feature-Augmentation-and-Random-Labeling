#!/usr/bin/env bash

EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN_TORCH=pretrained/resnet101-5d3b4d8f.pth
METHOD=$2 # choose one of baseline, FA, FARL, iFARL


BASE_WEIGHT=pretrained/coco_model_reset_surgery.pth
# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
            --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/defrcn_gfsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/${METHOD}/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH} --method ${METHOD}   \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/${METHOD}/defrcn_gfsod_r101_novel/tfa-like --shot-list 1 2 3 5 10 30  # surmarize all results