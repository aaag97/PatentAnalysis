import pickle
import torch, torchvision
import detectron2
# setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from cv2 import imshow
import argparse
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
from detectron2.utils.visualizer import ColorMode
import argparse
# setup_logger()
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import matplotlib.pyplot as plt

def classify_page(page_im, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to segement a given patent page
    Args:
    page_im - the image of the page to segment
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    gpu - boolean to indicate whether to use a gpu or not
    Returns:
    output - the output of the model (a dictionary containing an Instances object)
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    if not gpu:
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
    
    
def classify_patent(pages, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to segement a whole patent
    Args:
    pages - the images of the pages to segment
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    gpu - boolean to indicate whether to use a gpu or not
    Returns:
    outputs - a list of the outputs for each image (each output is a dictionary containing an Instances object)
    """
    outputs = [classify_page(pages[page_im_nb], config_path, weights_path, gpu) for page_im_nb in range(len(pages))]
    return outputs

def page_has_drawing(page_im, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to test whether a given patent page has a drawing
    Args:
    page_im - the image of the page to segment
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    gpu - boolean to indicate whether to use a gpu or not
    Returns:
    a boolean which is True if the page has a drawing according to the model and False otherwise
    """
    output = classify_page(page_im, config_path, weights_path, gpu)
    return 4 in [i.item() for i in output['instances'].pred_classes]

def patent_has_drawing(pages, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to test whether a given patent has a drawing
    Args:
    pages - the images of the pages of the patent
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    gpu - boolean to indicate whether to use a gpu or not
    Returns:
    a boolean which is True if the patent has a drawing according to the model and False otherwise
    """
    outputs = classify_patent(pages, config_path, weights_path, gpu)
    l = [[i.item() for i in output['instances'].pred_classes] for output in outputs]
    flatten = lambda list_: [item for sublist in list_ for item in sublist]
    return 4 in flatten(l)

def extract_text_pages(pages, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to extract the text pages of a given patent
    Args:
    pages - the images of the pages of the patent
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    gpu - boolean to indicate whether to use a gpu or not
    Returns:
    text_pages - the image files of the text pages
    indices - the indices of the text pages in the patent
    """
    text_pages = []
    indices = []
    for page_i in range(len(pages)):
        if not page_has_drawing(pages[page_i], config_path, weights_path, gpu):
            text_pages.append(pages[page_i])
            indices.append(page_i)
    return text_pages, indices
    
def extract_class(pages, class_, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to extract pages containing instance(s) a specific class
    Args:
    pages - the images of the pages of the patent
    class_ - the class the pages of which to extract
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    gpu - boolean to indicate whether to use a gpu or not
    Returns:
    imgs - the image files of the pages containing the class instance(s)
    """
    outputs = classify_patent(pages, config_path, weights_path, gpu)
    imgs = []
    for output_i in range(len(outputs)):
        for i in range(len(outputs[output_i]['instances'])):
            if outputs[output_i]['instances'][i].pred_classes.item() == class_:
                x1, y1, x2, y2 = [int(j.item()) for j in outputs[output_i]['instances'][i].pred_boxes.tensor[0]]
                imgs.append(pages[output_i][y1:y2,x1:x2])
    return imgs
    
def extract_imgs(pages, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to extract the figure pages of a given patent
    Args:
    pages - the images of the pages of the patent
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    gpu - boolean to indicate whether to use a gpu or not
    Returns:
    imgs - the image files of the pages containing the figure instance(s)
    """    
    return extract_class(pages=pages, class_=4, config_path=config_path, weights_path=weights_path, gpu=gpu)
    
def extract_text(pages, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to extract the text pages of a given patent
    Args:
    pages - the images of the pages of the patent
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    gpu - boolean to indicate whether to use a gpu or not
    Returns:
    imgs - the image files of the pages containing the text instance(s)
    """    
    return extract_class(pages=pages, class_=0, config_path=config_path, weights_path=weights_path, gpu=gpu)

def segment_and_save_result_img(pages, output_dir, page_names, config_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/detectron2/detectron_config/DLA_mask_rcnn_R_101_FPN_3x.yaml', weights_path='/Users/andrealphonse/Documents/UniStuff/MA/MA3/Classes/Patent Project/PubLayNet/new_finetuned_detectron2/non_sigil/resnet101/model_final.pth', gpu=False):
    """
    function to segment and save the segmented images of the pages
    Args:
    pages - the images of the pages of the patent
    output_dir - the directory we would like to save the segmented images in
    gpu - boolean to indicate whether to use a gpu or not
    config_path - the path to the detectron2 config yaml file
    weights_path - the path to a model
    Returns:
    outputs - a list of the outputs for each image (each output is a dictionary containing an Instances object)
    """  
    MetadataCatalog.get("patents_dataset").set(thing_classes=['text', 'title', 'list', 'table', 'figure', 'sigil'])
    patents_metadata = MetadataCatalog.get("patents_dataset")
    outputs = classify_patent(pages, config_path, weights_path, gpu)
    for im_index in range(len(pages)):    
        im = pages[im_index]
        output = outputs[im_index]
        v = Visualizer(im[:, :, ::-1],
                       metadata=patents_metadata, 
                       scale=0.5, 
                       instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        plt.figure(num=None, figsize=(9, 15), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.savefig('{}/{}.png'.format(output_dir, page_names[im_index]))
    return outputs

def get_ordered_images(folder):
    """
    function to get all images from a given folder
    Args:
    folder - the folder which contains the images
    Returns:
    images - the images in cv2 format
    img_names - the filenames of the images
    """
    imgs = []
    filenames = sorted(os.listdir(folder))
    img_names = []
    for filename_i in range(len(filenames)):
        filename = filenames[filename_i]
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            imgs.append(img)
            img_names.append(filename.split('.')[0])
    return imgs, img_names

def parse_args():
    """
    function to parse command line arguments
    Returns:
    args - an argspace object containing the arguments
    """
    parser=argparse.ArgumentParser(description='Segement and visualize patent pages.')
    parser.add_argument('-i', '--input', type=str, help='the path to a folder containing the images', required=True)
    parser.add_argument('-o', '--output', type=str, help='path to the output folder where the segmented images are to be stored', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='0 if you would like to use the available gpu and 1 otherwise', required=True)
    args = parser.parse_args()
    return args

def main(pages, output_path, page_names, gpu):
    if gpu == 0:
        gpu_b = False
    else:
        gpu_b = True
    segment_and_save_result_img(pages, output_dir=output_path, page_names=page_names, gpu=gpu_b)

if __name__ == "__main__":
    args = parse_args()
    input_path = args.input
    output_path = args.output
    gpu = args.gpu
    pages, page_names = get_ordered_images(folder=input_path)
    main(pages, output_path=output_path, page_names=page_names, gpu=gpu)
