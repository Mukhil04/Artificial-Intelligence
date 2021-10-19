# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    dictionary = {}
    final = []
    most_common = {}
    for sentence in train:
        for tuples in sentence:
            if tuples[0] not in dictionary:
                small_dict = {}
                small_dict[tuples[1]] = 1
                dictionary[tuples[0]] = small_dict
            else:
                if tuples[1] in dictionary[tuples[0]]:
                    dictionary[tuples[0]][tuples[1]] = dictionary[tuples[0]][tuples[1]] + 1
                else:
                    dictionary[tuples[0]][tuples[1]] = 1
            if tuples[1] not in most_common:
                most_common[tuples[1]] = 1
            else:
                most_common[tuples[1]] += 1
    abundance = {}
    unseen = max(most_common, key = most_common.get)
    for keys in dictionary.keys():
        abundance[keys] = (max(dictionary[keys], key = dictionary[keys].get), dictionary[keys][max(dictionary[keys], key = dictionary[keys].get)])
    for sentence in test:
        arr = []
        for word in sentence:
            if word == 'START':
                arr.append((word, 'START'))
            elif word == 'END':
                arr.append((word, 'END'))
            else:
                if word in abundance:
                    arr.append((word, abundance[word][0]))
                else:
                    arr.append((word, unseen))
        final.append(arr)
    return final
