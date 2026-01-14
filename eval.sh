#!/bin/bash 
set -eu

checkpoint=/node/yym/expriment/LExt/demo
gpuid=0

data_root=/node/yym/node/LExt/data/test

mix_scp=$data_root/mix_clean.scp 
spk1_scp=$data_root/ref.scp 
aux_scp=$data_root/auxs1.scp 

cal_sdr=1

python ./nnet/evaluate.py \
  --checkpoint $checkpoint \
  --gpuid $gpuid \
  --mix_scp $mix_scp \
  --ref_scp $spk1_scp \
  --aux_scp $aux_scp \
  --cal_sdr $cal_sdr \
> eval.log 2>&1

echo "eval done!"
