import pickle
import torch, torchvision
import detectron2
# setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from cv2 import imshow
import argparse
import glob
import multiprocessing as mp
import time
import cv2
import tqdm
from PIL import Image
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
# setup_logger()

from DetectronUtils import VisualizationDemo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import pytesseract

def classify_page(page_im, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth'):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.MODEL.DEVICE = 'cpu'
#     cfg.DATASETS.TEST = ("patents2_test", )
    cfg.MODEL.WEIGHTS = weights_path
    cfg.freeze()
#     model = build_model(cfg)
#     DetectionCheckpointer(model).load(weights_path)
    predictor = DefaultPredictor(cfg)
    output = predictor(page_im)
    return output
    # demo = VisualizationDemo(cfg)
    # predictions, visualized_output = demo.run_on_image(img)
    # visualized_output.save('Detectron2LayoutAnalysisOutput/3408_00000002')
    
    
def classify_patent(pages):
    outputs = [classify_page(page_im) for page_im in pages]
    return outputs

def page_has_drawing(page_im):
    output = classify_page(page_im)
    return 4 in [i.item() for i in output['instances'].pred_classes]

def patent_has_drawing(pages):
    outputs = classify_patent(pages)
    l = [[i.item() for i in output['instances'].pred_classes] for output in outputs]
    flatten = lambda list_: [item for sublist in list_ for item in sublist]
    return 4 in flatten(l)

def extract_text_pages(pages):
    text_pages = []
    indices = []
    for page_i in range(len(pages)):
        if not page_has_drawing(pages[page_i]):
            text_pages.append(pages[page_i])
            indices.append(page_i)
    return text_pages, indices
    
def extract_class(pages,class_):
    outputs = classify_patent(pages)
    imgs = []
    for output_i in range(len(outputs)):
        for i in range(len(outputs[output_i]['instances'])):
            if outputs[output_i]['instances'][i].pred_classes.item() == class_:
                x1, y1, x2, y2 = [int(j.item()) for j in outputs[output_i]['instances'][i].pred_boxes.tensor[0]]
                imgs.append(pages[output_i][y1:y2,x1:x2])
    return imgs
    
def extract_imgs(pages):
    return extract_class(pages=pages,class_=4)
    
def extract_text(pages):
    return extract_class(pages=pages,class_=0)
