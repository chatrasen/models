
#!/bin/bash



# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=~/tmp/wbc_morphle-models/resnet_v1_50

# Where the dataset is saved to.
DATASET_DIR=~/tmp/data/wbc_morphle


python get_eval_accuracies.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=wbc \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50
