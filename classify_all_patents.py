from patent_classification import patent_has_drawing
from regex_patent_classification import has_drawing_from_path, has_drawing
from ocrutils import OCR_from_pdf
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import pandas as pd
import argparse
import datetime
import glob
import numpy as np
import os
import warnings
from tqdm import tqdm

def pdf_to_images(patent_pdf_path):
    """
    function to convert a pdf file to images
    args:
    patent_pdf_path - the path to the pdf file
    returns:
    pages - a list of the images
    """
    pages = convert_from_path(patent_pdf_path)
    return pages

def dl_method_classification(patent_pdf_path, config_path, weights_path, gpu):
    """
    function to determine whether a pdf file of a patent contains a drawing according to the DL method
    args:
    patent_pdf_path - the path to the pdf file
    config_path - the path to the detectron2 model configuration file
    weights_path - the path to the detectron2 model (weights)
    gpu - whether a gpu is available or not (can be boolean or binary value)
    returns:
    does_it_have_drawing - True if patent contains a drawing and False otherwise
    """
    pages = pdf_to_images(patent_pdf_path)
    cv_pages = []
    for i in range(len(pages)):
        open_cv_image = np.array(pages[i]) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        cv_pages.append(open_cv_image)
    does_it_have_drawing = patent_has_drawing(cv_pages, config_path, weights_path, gpu)
    return does_it_have_drawing

def dl_method_add_classification(row, patents_path, config_path, weights_path, gpu):
    """
    function to add a "Has Drawing (according to DL method)" column to a row of an Excel file of patents
    args:
    row - a row in the given excel file
    patents_path - path to folder containing the patents
    config_path - the path to the detectron2 model configuration file
    weights_path - the path to the detectron2 model (weights)
    gpu - whether a gpu is available or not (can be boolean or binary value)
    returns:
    row - the same row with the extra column ("Has Drawing (according to DL method)")
    """
    date = datetime.datetime.strptime(row['Date'], '%d/%m/%Y')
    year = date.year
    year_range = [year, year - 1, year + 1, year - 2, year + 2, year - 3, year + 3, year - 4, year + 4]
    filenames = ['GB{}{:05d}A.pdf'.format(year_i, row['Number']) for year_i in year_range]
    file_paths = []
    for filename_i in range(len(filenames)):
        file_paths_i = glob.glob("{}/*/*/{}".format(patents_path, filenames[filename_i]), recursive = True)
        if file_paths_i != []:
            if os.path.exists(file_paths_i[0]):
                file_paths.append(file_paths_i[0])
    file_path = file_paths[0]
    does_it_have_drawing_b = dl_method_classification(file_path, config_path, weights_path, gpu)
    if does_it_have_drawing_b:
        row['Has Drawing (according to the DL method)'] = 'Yes'
    else:
        row['Has Drawing (according to the DL method)'] = 'No'
    return row

def dl_method_classification_all_patents(patents_excel_path, patents_path, output_filename, config_path, weights_path, gpu):
    """
    function to add a "Has Drawing (according to DL method)" column to an Excel file of patents
    args:
    patents_excel_path - the path to the Excel file containing the patent information
    patents_path - path to folder containing the patents
    output_filename - the name of the Excel file to output
    config_path - the path to the detectron2 model configuration file
    weights_path - the path to the detectron2 model (weights)
    gpu - whether a gpu is available or not (can be boolean or binary value)
    returns:
    row - the same row with the extra column ("Has Drawing (according to DL method)")
    """
    all_patents_gb_df = pd.read_excel(patents_excel_path)
    all_patents_gb_with_drawing_df = all_patents_gb_df.drop(columns=['Patentee', 'Subject', 'Class']).drop_duplicates()
    tqdm.pandas()
    all_patents_gb_with_drawing_df = all_patents_gb_with_drawing_df.progress_apply(lambda row: dl_method_add_classification(row, patents_path, config_path, weights_path, gpu), axis=1)
    all_patents_gb_df = all_patents_gb_df.merge(all_patents_gb_with_drawing_df, on=['Number', 'Date'])
    all_patents_gb_df.to_excel(output_filename)

def regex_method_classification(patent_pdf_path):
    """
    function to determine whether a pdf file of a patent contains a drawing according to the regex method
    args:
    patent_pdf_path - the path to the pdf file
    returns:
    does_it_have_drawing - True if patent contains a drawing and False otherwise
    """
    patent_str = OCR_from_pdf(patent_pdf_path)
    does_it_have_drawing = has_drawing(patent_str)
    return does_it_have_drawing is not None

def regex_method_add_classification(row, patents_path):
    """
    function to add a "Has Drawing (according to Regex method)" column to a row of an Excel file of patents
    args:
    row - a row in the given excel file
    patents_path - path to folder containing the patents
    returns:
    row - the same row with the extra column ("Has Drawing (according to Regex method)")
    """
    date = datetime.datetime.strptime(row['Date'], '%d/%m/%Y')
    year = date.year
    year_range = [year, year - 1, year + 1, year - 2, year + 2, year - 3, year + 3, year - 4, year + 4]
    filenames = ['GB{}{:05d}A.pdf'.format(year_i, row['Number']) for year_i in year_range]
    file_paths = []
    for filename_i in range(len(filenames)):
        file_paths_i = glob.glob("{}/*/*/{}".format(patents_path, filenames[filename_i]), recursive = True)
        if file_paths_i != []:
            if os.path.exists(file_paths_i[0]):
                file_paths.append(file_paths_i[0])
    file_path = file_paths[0]
    does_it_have_drawing_b = regex_method_classification(file_path)
    if does_it_have_drawing_b:  
        row['Has Drawing (according to the Regex method)'] = 'Yes'
    else:
        row['Has Drawing (according to the Regex method)'] = 'No'
    return row

def regex_method_classification_all_patents(patents_excel_path, patents_path, output_filename):
    """
    function to add a "Has Drawing (according to Regex method)" column to an Excel file of patents
    args:
    patents_excel_path - the path to the Excel file containing the patent information
    patents_path - path to folder containing the patents
    output_filename - the name of the Excel file to output
    returns:
    row - the same row with the extra column ("Has Drawing (according to Regex method)")
    """
    all_patents_gb_df = pd.read_excel(patents_excel_path)
    all_patents_gb_with_drawing_df = all_patents_gb_df.drop(columns=['Patentee', 'Subject', 'Class']).drop_duplicates()
    tqdm.pandas()
    all_patents_gb_with_drawing_df = all_patents_gb_with_drawing_df.progress_apply(lambda row: regex_method_add_classification(row, patents_path), axis=1)
    all_patents_gb_df = all_patents_gb_df.merge(all_patents_gb_with_drawing_df, on=['Number', 'Date'])
    all_patents_gb_df.to_excel(output_filename)

def parse_args():
    """
    function to parse command line arguments
    Returns:
    args - an argspace object containing the arguments
    """
    parser=argparse.ArgumentParser(description='Add an extra column to determine if patents have drawings or not according to one of two methods.')
    parser.add_argument('-i', '--input', type=str, help='the path to the excel file containing the information regarding the patents', required=True)
    parser.add_argument('-f', '--patentfolder', type=str, help='path to the folder containing the pdf documents of the patents', required=True)
    parser.add_argument('-o', '--output', type=str, help='path to the output file that will be created with the extra column', required=True)
    parser.add_argument('-m', '--method', type=str, help='method to be used\nenter \'DL\' to use deep learning method or \'Regex\' to use Regex method', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='0 if you would like to use the available gpu and 1 otherwise (only used when deep learning method is picked, irrelevant otherwise)')
    parser.add_argument('-c', '--config', type=str, help='path to detectron2 model configuration (only used when deep learning method is picked, irrelevant otherwise)')
    parser.add_argument('-w', '--weights', type=str, help='path to detectron2 model weights (.pth file) (only used when deep learning method is picked, irrelevant otherwise)')
    args = parser.parse_args()
    if args.method.lower() == 'dl' and (args.config is None or args.weights is None or args.gpu is None):
        parser.error("Using the DL method requires --weights, --config and --gpu.")
    return args

def main(args):
    warnings.filterwarnings("ignore")
    method = args.method
    patents_excel_path = args.input
    output_excel_path = args.output
    patents_path = args.patentfolder
    if method.lower() == 'dl':
        config_path = args.config
        weights_path = args.weights
        gpu = args.gpu
        if gpu == 0:
            gpu_b = False
        else:
            gpu_b = True
        dl_method_classification_all_patents(patents_excel_path, patents_path, output_excel_path, config_path, weights_path, gpu_b)
    elif method.lower() == 'regex':
        regex_method_classification_all_patents(patents_excel_path, patents_path, output_excel_path)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

    
#     pages, page_names = get_ordered_images(folder=input_path)
#     main(pages, output_path=output_path, page_names=page_names, config_path=config_path, weights_path=weights_path, gpu=gpu)