# Capstone Projects: Cyberbullying Prediction in Social Media

Predicting Cyberbullying in Social Media (Instagram) using Machine Learning Techniques. This is an overview of my final original capstone project for the EPFL extension school "Applied Data Science: Machine Learning" program. 

Key Question: Can we use social media data like Instagram comment to predict cyberbullying. The data used in this project was made available by the University of Colorado per request for the purpose of this project. http://www.cucybersafety.org/home/publications [https://arxiv.org/pdf/1508.06257v1.pdf]

**Models**

- **Logistic Regression:** typical classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables. This will also be the baseline against which I compare the performance of the other algorithms.
- **Naïve Bayes:** a family of probabilistic algorithms that uses Bayes’s Theorem to predict the category of a text. It is a very popular method of text classification and uses word frequency as its features. In the current set of problem, for determining a sentiment polarity into a positive or a negative sentiment, the Naïve Bayes is considered as one of the best algorithm that can be used for a binomial classification problem.
- **XGBoost:** Extreme Gradient Boosting (xgboost) is an advanced implementation of gradient boosting algorithm. It has both linear model solver and tree learning algorithms. Its ability to do parallel computation on a single machine makes it extremely fast. It also has additional features for doing cross validation and finding important variables. 
- **Neural Nets:** Recently Convolutional Neural Networks (CNNs) models have proven remarkable results for text classification and sentiment analysis.


In order to run the notebooks, the following files are needed (separate download is required, the size is too big to upload via GitHub):

- crawl-300d-2M.vec (from: https://fasttext.cc/docs/en/english-vectors.html)
- glove.twitter.27B.200d.txt (from: https://nlp.stanford.edu/projects/glove/)

All the code was implemented as Jupyter Notebooks (Python 3.6). The following is a description of each one of the available files:

- EDA.ipynb:
Data preprocessing (cleaning, visualization and understanding of the issues and limitations of the data set).
- Feature Engineering TF-IDF.ipynb:
Extract TF-IDF features using TF-IDF Vectorizer.
- Feature Engineering Word2Vec.ipynb:
Create word embeddings using Gensim and create word vectors for predictive modelling.
- Feature Engineering Glove/FastText.ipynb:
Create word embeddings using pretrained GloVe and FastText and create word vectors for predictive modelling.
- Logistic Regression.ipynb:
Apply the logistic regression model to predict cyberbullying. 
- Naive Bayes.ipynb:
Apply the naive Bayes model to predict cyberbullying.
- XGBoost.ipynb:
Apply the extreme gradient boosting model to predict cyberbullying. 
- XGBoost imbalanced.ipynb
Apply the extreme gradient boosting model on imbalanced data to predict cyberbullying. 
- XGBoost GridSearchCV.ipynb
Apply the extreme gradient boosting model with GridSearchCV to predict cyberbullying. .
- ConvNets Sampling.ipynb
Apply convnets and compare different methods to deal with data imbalance. 
- ConvNets Embeddings.ipynb
Apply convnets and compare different ways to apply embeddings. 
- ConvNets TF-IDF.ipynb
Apply convnets to features extracted using TF-IDF vectorizer. 
- ConvNets Word2Vec.ipynb
Apply convnets to features extracted using Gensim's Word2Vec. 
- Results.ipynb
Finally, this script compares and discussed the results, as well as potential ideas for further investigation.