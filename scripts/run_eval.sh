# ~/semantic segmentation

export PYTHONPATH=$(pwd)
python eval.py --model_module "models.unet" \
               --model_config_file "models/unet.json" \
               --model_weights_file "tmp/checkpoints/0011_0.9038_best.pth" \
               --data_root "/home/cotai/giang/datasets/VOC-2012" \
               --label_map_file "/home/cotai/giang/datasets/VOC-2012/label_map.json" \
               --device "cuda:1"