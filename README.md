Spam Review Detection using SOM-CNN

Prerequisites to run the codes:

•	Pretrained GloVe-300 dimension word representation vectors which can be downloaded from https://nlp.stanford.edu/projects/glove/. 

•	Installing or copying the SRD_minisom.py in the same folder as the algorithms. SRD_minisom.py, included here, is our modified version of the MiniSom.py library originally retrieved from https://test.pypi.org/project/MiniSom/. 

•	Using python, tensorflow version of 2.2.0 and keras version of 2.3.1.


The code can be run following these steps:

Step 1: Preprocessing
Cleaning the reviews, calculating the TF/IDF and constructing word embedding matrix using preprocessing.py.

Step 2 : SOM Clustering of words
Clustering the vocabulary and constructing SOM image map by importing SRD_minisom.py library using the Clustering_words_SOM.py

Step 3 : Reviews image construction and CNN classification
Constructing the review images through SOM image map, training and classification by using Constructing_images_classification.py

We used Google Colab environment to perform our experiments.

