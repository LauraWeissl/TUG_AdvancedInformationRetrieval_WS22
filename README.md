# FBI - FakeNews BERT Inspector

Project for the course Advanced Information Retrieval. 
This project includes scripts for the tasks of text analysis and fake news detection.

Authors: 
- Sebastian Weidinger
- Laura Weißl 

## Getting Started 

### Data Analysis

The first part of the project is a text analysis of *Fake* and *True* labelled news articles. This analysis will provide information and generate plots about differences regarding the following characteristics:

- Total number of words in text
- Most frequent words in text
- Most frequent named entities (Persons, Organizations, Geopolitical entities) in text
- Topic of text
- Sentiment of Text
    - Negative/Neutral/Positive 
    - Emotion (sadness, joy, love, anger, fear and surprise)

The implementation of the *news_analysis.ipynb* was done with Kaggle, however, the file can also be run with Jupyter (path for dataset must be adapted).

#### Models for Sentiment Analysis
The sentiment analysis was carried out with the following two models from Hugging Face:

- [Twitter-roBERTa-base for Sentiment Analysis](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [Distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)

### Classification
To train and test the detection model execute the pipeline in main.py with the options: 
- --test
  - for testing the model
- --train
  - for training the model
- --load_model
  - for loading an existing model

For example, to train, test and load a pretrained model execute: 

```console
python main.py --train --test --load_model
```

The results of the training and testing are saved in the locations "./results" and "./plots".
A pretrained model can be found in the directory "./models/final".

To create sentiment classifications and to analyze the prediction results execute analyzer.py.

### Requirements

The required imports for Data Analysis and Classification can be found in the *requirements.txt* file.

## Dataset

The used dataset has to be downloaded from Kaggle: 

- [Kaggle Dataset Misinformation](https://www.kaggle.com/datasets/stevenpeutz/misinformation-fake-news-text-dataset-79k)

Details about the dataset can be found on Kaggle. 
The files of the dataset should be placed into the directory "./archive/archive". 