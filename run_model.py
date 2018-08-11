from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import csv

import process
import model


train_x, valid_x, train_y, valid_y,trainDF = process.prep_data()

xtrain_count, xvalid_count = process.countVec(train_x, valid_x, trainDF)

xtrain_tfidf, xvalid_tfidf, xtrain_tfidf_ngram, xvalid_tfidf_ngram, xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars = process.tf_idf(train_x, valid_x, trainDF)

embedding_matrix = process.word_emb(trainDF, train_x, valid_x)

DF = process.apply_pos(trainDF)


def getChoice():
    print("\nMenu\n(1)Naive Bayes Classifier\n(2)Linear Classifier\n(3)SVM\n(Q)uit")
    choose=raw_input(">>> ")
    choice=choose.lower()

    return choice

choice = getChoice()

def naive_bayes():
    # Naive Bayes on Count Vectors
    accuracy = model.train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
    print "NB, Count Vectors: ", accuracy

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = model.train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
    print "NB, WordLevel TF-IDF: ", accuracy

    # Naive Bayes on Ngram Level TF IDF Vectors
    accuracy = model.train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print "NB, N-Gram Vectors: ", accuracy

    # Naive Bayes on Character Level TF IDF Vectors
    accuracy = model.train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print "NB, CharLevel Vectors: ", accuracy


def linear_classifier():
        # Linear Classifier on Count Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
    print "LR, Count Vectors: ", accuracy

    # Linear Classifier on Word Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
    print "LR, WordLevel TF-IDF: ", accuracy

    # Linear Classifier on Ngram Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print "LR, N-Gram Vectors: ", accuracy

    # Linear Classifier on Character Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print "LR, CharLevel Vectors: ", accuracy

def svm():
    accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print "SVM, N-Gram Vectors: ", accuracy



while choice!="q":
    if choice=="1":
        naive_bayes()
    elif choice=="2":
        linear_classifier()
    elif choice=="3":
        svm()

    else:
        print("Invalid choice, please choose again")
        print("\n")

    choice = getChoice()
