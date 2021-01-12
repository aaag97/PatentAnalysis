from multiprocessing import Pool
import os
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import cv2 
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
import re
import PyPDF2 as pyPdf
import time
import random
from PIL import Image


def OCR_Pages(pages):
    patent_str = ''
    
    for page_index in range(len(pages)):
        str_from_img = pytesseract.image_to_string(pages[page_index], config='--psm 1')
        patent_str = '{}{}'.format(patent_str, str_from_img)
    return patent_str

def OCR_GB_patent(patent_pdf_path_index, patent_list, dest="/Volumes/Non-Backup_Files/GB-patents/MachineReadableBaseline/"):
    patent_str = ''
    patent_pdf_path = patent_list[patent_pdf_path_index]
    patent_name = patent_pdf_path.split('/')[-1][:-4]
    imgs = convert_from_path(patent_pdf_path)
    for img_index in range(len(imgs)):
        img = imgs[img_index]
        str_from_img = pytesseract.image_to_string(img)
        patent_str = '{}\n{}'.format(patent_str, str_from_img)
    with open('{}{}.txt'.format(dest, patent_name), "w") as text_file:
        text_file.write("%s" % patent_str)
    if(patent_pdf_path_index % 1000 == 0):
        print('finished {}'.format(patent_pdf_path_index))
        
        
def OCR_US_patent(patent_nb_index, patent_dict, patent_list, dest="/Volumes/Non-Backup_Files/US-patents/MachineReadableBaseline/"):
    patent_str = ''
    patent_nb = patent_list[patent_nb_index]
    
    for img_index in range(len(patent_dict[patent_nb])):
        img = Image.open(patent_dict[patent_nb][img_index])
        str_from_img = pytesseract.image_to_string(img)
        patent_str = '{}\n{}'.format(patent_str, str_from_img)
        
    with open('{}/{}.txt'.format(dest, patent_nb), "w") as text_file:
        text_file.write("%s" % patent_str)