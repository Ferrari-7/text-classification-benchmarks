# Assignment 2 - Text classification benchmarks

This repository contains code which will use ```scikit-learn``` to train binary classification models on a dataset of news articles labelled as either fake and real. The dataset can be found found on Kaggle via this [link](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news)

This repository contains *two different scripts*. One script trains a logistic regression classifier on the data; the second script trains a neural network on the same dataset. Both scripts does the following:

- Saves a classification report to the folder called ```out```
- Saves the trained model and vectorizer to the folder called ```models```


## User instructions

1. Install necessary packages using the setup script like so: 

```bash setup.sh```

2. Run the script which trains the logistic regression classifier by navigating to the ```src``` folder:

```python logreg.py```

3. Run the script which trains the neural network by navigating to the ```src``` folder: 

```python mlp_clf.py```


## Discussion of results

This is the classification report for the **logistic regression classifier**: 

                precision    recall  f1-score   support

    FAKE            0.86      0.74      0.79       628
    REAL            0.78      0.88      0.82       639

    accuracy                            0.81      1267
    macro avg       0.82      0.81      0.81      1267
    weighted avg    0.81      0.81      0.81      1267

And for the **neural network**: 

              precision    recall  f1-score   support

    FAKE            0.86      0.73      0.79       628
    REAL            0.77      0.88      0.82       639

    accuracy                            0.81      1267
    macro avg       0.81      0.80      0.80      1267
    weighted avg    0.81      0.81      0.80      1267

To my initial surprise, the logistic regression classifier seem to be outperforming the multilayer perceptron classifier (although just narrowly). But perhaps this makes sense since logistic regression is particularly suited for binary classification. The weighed avg for the f1-scores is just .01 higher for the logistic regression classifier. 

When I used the same type of models on image data with multible categories, the f1-scores were more than half as low. But these more simple classification models seem to work well on binary classification of text data. 

