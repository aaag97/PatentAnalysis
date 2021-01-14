# PatentAnalysis

Patents have long been granted by governments in order to award innovation. They are given when a patent specification is filed for an invention. This specification typically includes a detailed explanation as well as figures to illustrate the breakthrough which was reached by the innovator. It is hence a key medium to communicate innovation.

In this study, we aim to improve our understanding of patents as well as our approach to analyzing these documents in a digital fashion. To this end, we focus on a corpus of American and British patents. After testing several methods for drawing detection on the documents, we then move on to digitizing the them through OCR and analyzing the evolution of certain linguistic aspects of patent documents.

In order to use the notebooks, the data is required. However, the python files may be used as follows. To this end the repository should be cloned locally. Below are some instructions in regards to the usage of the python files.

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
python patent_classification.py -i <input> -o <output> -g <gpu>
```
where `<input>` is the path to a directory containing the photos to be segmented, `<output>` is the directory to store the segmentation results and `<gpu>` should be `0` if the GPU is to be used and `1` otherwise.
It should be noted that `patent_classification.py` contains many useful function for patent page segmentation which could be used. These include:
* _classify\_page_ which segments a page.
* _classify\_patent_ which segments a patent.
* _page\_has\_drawing_ which evaluates whether a page contains a figure.
* _patent\_has\_drawing_ which evaluates whether a patent contains a drawing.
* _extract\_text\_pages_ which extracts pages which contain text from a given patent.

The training for the detectron2 model which was used for image detection was done in a [Google Colaboratory notebook](https://colab.research.google.com/drive/1JKKf8BoSE0_t7DOonMmlUoeTZ9334R8X?usp=sharing "LAYOUT ANALYSIS ON COLAB (DETECTRON2 TRAINING AND EVAL RESULTS)"). The Tesseract OCR process was also done in a [Google Colaboratory notebook](https://colab.research.google.com/drive/1nNmg5PLxYFgmxq9fambtWQkaHDjfdEos?usp=sharing "OCR PROCESS ON COLAB").

## NLP Analysis

In order to use the nlp_analysis.py file, the following command should be used - 
```
python nlp_analysis.py -i <input> -o <output> -a <all_refs>
```
where `<input>` is the path to the text file to be analyzed, `<output>` is the path to a json file to be created so that the results of the analysis should be stored and `<all_refs>` should be 1 if all pronouns find using the Spacy library should be returned (not recommended if there are spelling errors in the text file) and 0 otherwise.