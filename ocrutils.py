#deprecated
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

def OCR_from_pdf(patent_pdf_path):
    """
    function to OCR the pages of a patent given the path to a pdf of the patent
    Args:
    patent_pdf_path - path to pdf
    Returns:
    patent_str - a string of the all the pages after the OCR
    """
    patent_str = ''
    pages = convert_from_path(patent_pdf_path)
    for page_index in range(len(pages)):
        page = pages[page_index]
        str_from_page = pytesseract.image_to_string(page, config='--psm 1')
        patent_str = '{}{}'.format(patent_str, str_from_page)
    return patent_str

def OCR_Pages(pages):
    """
    function to OCR the pages given as an argument
    Args:
    pages - the list of pages in image form
    Returns:
    patent_str - a string of the all the pages after the OCR
    """
    patent_str = ''
    for page_index in range(len(pages)):
        str_from_img = pytesseract.image_to_string(pages[page_index], config='--psm 1')
        patent_str = '{}{}'.format(patent_str, str_from_img)
    return patent_str

def OCR_GB_patent(patent_pdf_path_index, patent_list, dest="/Volumes/Non-Backup_Files/GB-patents/MachineReadableBaseline/"):
    """
    function to OCR a Brtish patent
    Args:
    patent_pdf_path_index - the index of the patent number we want to OCR
    patent_list - a list of patent number
    dest - folder to use in which to store output
    Returns:
    None, writes text file in dest
    """
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
    """
    function to OCR an American patent
    Args:
    patent_nb_index - the index of the patent number in patent_list
    patent_dict - a dictionary from patent numbers to a list of the paths pointing to the page images of the patent numbers
    patent_list - a list of patent numbers
    dest - folder to use for storage of the resulting text file
    Returns:
    None, writes text file in dest
    """
    patent_str = ''
    patent_nb = patent_list[patent_nb_index]
    
    for img_index in range(len(patent_dict[patent_nb])):
        img = Image.open(patent_dict[patent_nb][img_index])
        str_from_img = pytesseract.image_to_string(img)
        patent_str = '{}\n{}'.format(patent_str, str_from_img)
        
    with open('{}/{}.txt'.format(dest, patent_nb), "w") as text_file:
        text_file.write("%s" % patent_str)