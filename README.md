# PatentAnalysis

Patents have long been granted by governments in order to award innovation. They are given when a patent specification is filed for an invention. This specification typically includes a detailed explanation as well as figures to illustrate the breakthrough which was reached by the innovator. It is hence a key medium to communicate innovation.

In this study, we aim to improve our understanding of patents as well as our approach to analyzing these documents in a digital fashion. To this end, we focus on a corpus of American and British patents. After testing several methods for drawing detection on the documents, we then move on to digitizing the them through OCR and analyzing the evolution of certain linguistic aspects of patent documents.

In order to use the notebooks, the data is required. However, the python files may be used as follows. To this end the repository should be cloned locally. Below are some instructions in regards to the usage of the python files. These instrcutions may be followed after installing all the libraries mentioned in the _requirements.txt_ file.

## Drawing Detection

### The Regex Method
In order to use the Regex Method to evaluate whether a given patent contains a figure or not - 

```
python regex_patent_classification.py -i <input> -o <output>
```
such that `<input>` is the path to a directory containing the patent pages as images and `<output>` is a path to a text file where the result will be written (either "This patent has one or more figures!" or "This patent does not have any figures!").

### The Deep Learning Method

In order to showcase the segmentation of a given patent, the following command could be used -

```
python patent_classification.py -i <input> -o <output> -g <gpu> -c <config> -w <weights>
```
where `<input>` is the path to a directory containing the photos to be segmented, `<output>` is the directory to store the segmentation results and `<gpu>` should be `0` if the GPU is to be used and `1` otherwise, `<config>` is the path to the detectron2 model yaml configuration file and `<weights>` is the path to the .pth weights.
It should be noted that `patent_classification.py` contains many useful function for patent page segmentation which could be used. These include:
* _classify\_page_ which segments a page.
* _classify\_patent_ which segments a patent.
* _page\_has\_drawing_ which evaluates whether a page contains a figure.
* _patent\_has\_drawing_ which evaluates whether a patent contains a drawing.
* _extract\_text\_pages_ which extracts pages which contain text from a given patent.


## NLP Analysis

In order to use the nlp_analysis.py file, the following command should be used - 
```
python nlp_analysis.py -i <input> -o <output> -a <all_refs>
```
where `<input>` is the path to the text file to be analyzed, `<output>` is the path to a json file to be created so that the results of the analysis should be stored and `<all_refs>` should be 1 if all pronouns find using the Spacy library should be returned (not recommended if there are spelling errors in the text file) and 0 otherwise.

The training for the detectron2 model which was used for image detection was done in a [Google Colaboratory notebook](https://colab.research.google.com/drive/1JKKf8BoSE0_t7DOonMmlUoeTZ9334R8X?usp=sharing "LAYOUT ANALYSIS ON COLAB (DETECTRON2 TRAINING AND EVAL RESULTS)"). The Tesseract OCR process was also done in a [Google Colaboratory notebook](https://colab.research.google.com/drive/1nNmg5PLxYFgmxq9fambtWQkaHDjfdEos?usp=sharing "OCR PROCESS ON COLAB").

## Script to add "Has Drawing" column

The scipt to add a column to an excel file indicating whether the patent has a drawing or not is called classify_all_patents.py. In order to use it the following command must be used:

```
python3 classify_all_patents.py -i <excel_input> -f <folder> -o <output> -m <method> 
```
where 
* `<excel_input>` is the path to  the Excel file (this file should contain a column "Number" for the patent number as well as a column "Date" where the date of the filing of the patents is given in the format "day/month/year" and a column "Pantentee" for the patentees),
* `<folder>` is the path to the folder containing the patents (the program assumes that the patents are organized in a hierarchy such that the PDF files are three levels below this folder, e.g. this folder should be the Pre1900 and the PDF files should be contained in subfolders of the each year folder in the Pre1900 folder),
* `<method>` is either `dl` if the deep learning method is to be used or `regex` if the regex method is to be used. In order to use the deep learning method, three additional arguments must be specified:
    * `-g <gpu>` where `<gpu>`  is 0 if there are no GPU's available and 1 otherwise,
    * `-c <config>` where `<config>` is the path to the Detectron2 configuration file,
    * `-w <weights>` where `<weights>` should be the path to the Detectron2 model to be used for image segmentation.
    
For instance if you would like to use the regex method, the command should resemble -
```
python3 classify_all_patents.py -i <excel_input> -f <folder> -o <output> -m <method> 
```
- and if you would like to use the deep learning method, the command should look like -
```
python3 classify_all_patents.py -i <excel_input> -f <folder> -o <output> -m <method> -g <gpu> -c <config> -w <weights>
```
