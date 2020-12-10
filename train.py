from detectron2.data.datasets import register_coco_instances
from detectron2.config.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.engine.defaults import DefaultTrainer

DATA_ROOT = '/media/how/How/Class/deep_learning/cs-t0828-2020-hw3'
register_coco_instances('tiny-pascal', {}, DATA_ROOT + '/dataset/pascal_train.json', DATA_ROOT + '/dataset/train_images/')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = DATA_ROOT + '/R-50.pkl'
cfg.DATASETS.TRAIN = ('tiny-pascal',)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001

trainer = DefaultTrainer(cfg)
trainer.resume_or_load()
trainer.train()
