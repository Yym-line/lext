#!/bin/bash 
set -eu
checkpoint=/node/yym/expriment/LExt/demo
gpuid=0
data_root=/node/yym/node/dataset/kws

mix_scp=$data_root/mix/farrever5.scp
aux_scp=$data_root/enroll/farrever5_p45.scp
#aux_scp=/node/yym/node/dataset/concan_kws/cosy_farrever10.scp
#aux_scp=/node/yym/expriment/indextts/kws_test/farrever10_p45_030.scp 

fs=16000
dump_dir=/node/yym/expriment/LExt/onlymix_kws1/farrever5

python ./nnet/separate.py \
  --checkpoint $checkpoint \
  --gpuid $gpuid \
  --mix_scp $mix_scp \
  --aux_scp $aux_scp \
  --fs $fs \
  --dump-dir $dump_dir \
 > separate.log 2>&1

echo "Separate done!"
