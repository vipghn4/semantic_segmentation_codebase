# ~/semantic segmentation

export PYTHONPATH=$(pwd)
python eval.py --model_module "models.deeplabv3" \
               --model_config_file "models/deeplabv3.json" \
               --model_weights_file "tmp/checkpoints/0049_0.7256.pth" \
               --data_root "/home/cotai/giang/datasets/VOC-2012" \
               --label_map_file "/home/cotai/giang/datasets/VOC-2012/label_map.json" \
               --device "cuda:1"\
               --save_dir "tmp/eval"