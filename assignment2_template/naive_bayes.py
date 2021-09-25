# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

from collections import Counter
import math

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    ham = {}
    spam = {}
    total_number_of_words_spam = 0
    total_number_of_words_ham = 0
    unique_words_ham = 0
    unique_words_spam = 0
    for i in range (len(train_set)):
        a = Counter(train_set[i])
        if train_labels[i] == 1:                                                    # Ham
            total_number_of_words_ham = total_number_of_words_ham + len(train_set(i))
            for element in set(a):
                if ham.get(element, 0) == 0:
                    ham[element] = a[element]
                    unique_words_ham = unique_words_ham + 1
                else:
                    ham[element] = ham[element] + a[element]
        else:                                                                       # Spam
            total_number_of_words_spam = total_number_of_words_spam + len(train_set(i))
            for element in set(a):
                if spam.get(element, 0) == 0:
                    spam[element] = a[element]
                    unique_words_spam = unique_words_spam + 1
                else:
                    spam[element] = spam[element] + a[element]
    
    for key in ham:
        ham[key] = ham[key]/total_number_of_words_ham
    for key in spam:
        spam[key] = spam[key]/total_number_of_words_spam 

    labels = []
    total_word_likelihood_ham = 0
    total_word_likelihood_spam = 0
    for i in range(len(dev_set)):
        for element in dev_set[i]:
            if element in ham.keys:
                total_word_likelihood_ham = total_word_likelihood_ham + math.log(ham[element])
            else:
                total_word_likelihood_ham = total_word_likelihood_ham + math.log((smoothing_parameter)/(len(element) + (smoothing_parameter * len(set(element)))))
            if element in spam.keys:
                total_word_likelihood_spam = total_word_likelihood_spam + math.log(spam[element])
            else:
                total_word_likelihood_spam = total_word_likelihood_spam + math.log((smoothing_parameter)/(len(element) + (smoothing_parameter * len(set(element)))))
        posterior_ham = pos_prior * total_word_likelihood_ham
        posterior_spam = (1 - pos_prior) * total_word_likelihood_spam
        if posterior_ham >= posterior_spam:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels
    