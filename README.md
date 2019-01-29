# Sentiment-Analysis
NLP, movie reviews dataset

The purpose of this project is to build a machine learning model using keras because sklearn models did not works well on this dataset.
The first approach was using nltk and SVM to build a machine learning model, but the result was not good. I believed the reason that sklearn model did not works good was because the position of the word matters in this case. So I decided to try keras LSTM model with pre-trained embedding layer. The embedding layer will save words in a dense vector that remeber the words' position, and LSTM's ability to forget which outstanding the performance of other models when fitting a large size of dataset.
