import os
import re
import argparse
from PIL import Image
from ocrutils import OCR_Pages

def has_drawing_from_path(text_path):
    """
    function to evaluate whether a patent has a drawing using regex
    Args:
    text_path - path towards text file
    Returns:
    boolean indicating whether the patent has a drawing or not according to the regex method
    """
    with open(text_path, "r") as text_file:
        text_file_str = text_file.read()
    return re.search('[Ff][Ii][Gg]\.[0-9]+', text_file_str) or re.search('[Ff][Ii][Gg]\. [0-9]+', text_file_str) or re.search('drawing', text_file_str)

def has_drawing(text_file_str):
    """
    function to evaluate whether a patent has a drawing using regex
    Args:
    text - string of patent
    Returns:
    boolean indicating whether the patent has a drawing or not according to the regex method
    """
    return re.search('[Ff][Ii][Gg]\.[0-9]+', text_file_str) or re.search('[Ff][Ii][Gg]\. [0-9]+', text_file_str) or re.search('drawing', text_file_str)

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
        img = Image.open(os.path.join(folder,filename))
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
    parser.add_argument('-o', '--output', type=str, help='path to the output text file', required=True)
    args = parser.parse_args()
    return args

def main(pages, output_path):
    text = OCR_Pages(pages)
    if has_drawing(text):
        print('This patent has one or more figures!')
        with open(output_path, 'w') as f:
            f.write('This patent has one or more figures!')
    else:
        print('This patent does not have any figures!')
        with open(output_path, 'w') as f:
            f.write('This patent does not have any figures!')

if __name__ == "__main__":
    args = parse_args()
    input_path = args.input
    output_path = args.output
    pages, page_names = get_ordered_images(folder=input_path)
    main(pages, output_path=output_path)