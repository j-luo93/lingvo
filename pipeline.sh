gpu=$1
td=$2
model=$3
CUDA_VISIBLE_DEVICES=$gpu bazel-bin/lingvo/trainer --mode sync --run_locally gpu --model lm.ptb_word_level.$model --logdir train/$td/ --logtostderr --worker_gpus 1  &> /dev/null &
CUDA_VISIBLE_DEVICES=$((gpu+1)) bazel-bin/lingvo/trainer --mode sync --run_locally gpu --model lm.ptb_word_level.$model --logdir train/$td/ --logtostderr --job evaler_dev,evaler_test --evaler_gpus  1  &> /dev/null &
