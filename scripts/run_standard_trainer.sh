# ~/semantic segmentation

export PYTHONPATH=$(pwd)
python trainers/standard_trainer.py --model_module "models.deeplabv3" \
                                    --model_config_file "models/deeplabv3.json" \
                                    --data_root "/home/cotai/giang/datasets/VOC-2012" \
                                    --label_map_file "/home/cotai/giang/datasets/VOC-2012/label_map.json" \
                                    --device "cuda:1"