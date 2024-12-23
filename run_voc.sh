#!/usr/bin/env bash

EXP_NAME=$1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN_TORCH=pretrained/resnet101-5d3b4d8f.pth
SPLIT_ID=$2
METHOD=$3 # choose one of baseline, FA, FARL, iFARL


BASE_WEIGHT=pretrained/voc_split${SPLIT_ID}_model_reset_surgery.pth
# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10   # if final, 10 -> 1 2 3 5 10
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/${METHOD}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH} --method ${METHOD}         \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like --shot-list 1 2 3 5 10  # surmarize all results