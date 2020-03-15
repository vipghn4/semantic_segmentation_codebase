# ~/semantic segmentation

export PYTHONPATH=$(pwd)
python trainers/standard_trainer.py --model_module "models.unet" \
                                    --model_config_file "models/unet.json" \
                                    --data_root "/home/cotai/giang/datasets/VOC-2012" \
                                    --label_map_file "/home/cotai/giang/datasets/VOC-2012/label_map.json"