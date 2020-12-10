import cv2
import json
import torch
from tqdm import tqdm
from utils import binary_mask_to_rle
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = 'output/model_0259999.pth'
    cfg.DATASETS.TRAIN = ('tiny-pascal',)
    predictor = DefaultPredictor(cfg)
    DATA_ROOT = '/media/how/How/Class/deep_learning/cs-t0828-2020-hw3'
    result = []

    with open('dataset/test.json') as f:
        test_file = json.load(f)
        name_list = [test_file['categories'][i]['name'] for i in range(20)]
        register_coco_instances('tiny-pascal', {}, DATA_ROOT + '/dataset/pascal_train.json', DATA_ROOT + '/dataset/train_images/')
        MetadataCatalog.get('tiny-pascal').set(thing_classes=name_list)
    
        for img_data in tqdm(test_file['images']):
            img_name = img_data['file_name']
            img = cv2.imread('dataset/test_images/' + img_name)
            output = predictor(img)
            instance = output['instances']

            # for visualize result.
            '''
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(instance.to("cpu"))
            cv2.imshow('output', out.get_image()[:, :, ::-1])
            print(output)
            k = cv2.waitKey(0)
            if k == ord('q'):
                exit()
            '''
            # for output json file.
            pred_classes = instance.pred_classes
            pred_masks = instance.pred_masks
            pred_scores = instance.scores
            
            for i in range(len(pred_classes)):
                pred = {}
                pred['image_id'] = img_data['id']
                pred['score'] = float(pred_scores[i])
                pred['category_id'] = int(pred_classes[i]+1)
                pred['segmentation'] = binary_mask_to_rle(pred_masks[i, :, :].cpu().numpy())
                result.append(pred)

    with open("0856138.json", "w") as f:
        json.dump(result, f)
            
