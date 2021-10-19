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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
import numpy as np

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    set1 = set()
    set2 = set()
    for sentences in train:
        for pair in sentences:
            set1.add(pair[1])
            set2.add(pair[0])
    set1.remove("END")
    set2.remove("START")
    set2.remove("END")
    transition = {}
    for item in set1:
        temp = {}
        temp1 = {}
        item_count = 0
        for i in range(len(train)):
            for j in range(len(train[i])-2):
                if train[i][j][1] == item:
                    item_count += 1
                    if train[i][j+1][1] in temp:
                        temp[train[i][j+1][1]] += 1
                    else:
                        temp[train[i][j+1][1]] = 1
        for element in set1:
            if element in temp:
                temp1[element] = math.log((temp[element] + 0.1)/(item_count + (0.1 * (len(set1) + 2))))
            else:
                temp1[element] = math.log(0.1/(item_count + (0.1 * (len(set1) + 2))))
        transition[item] = temp1
    initial = transition["START"]
    emission = {}
    words = {}
    tags = {}
    for i in range(len(train)):
        for j in range(len(train[i])):
            if train[i][j][1] == 'START' or train[i][j][1] == 'END':
                continue
            if train[i][j][1] in emission:
                tags[train[i][j][1]] += 1
                if train[i][j][0] in emission[train[i][j][1]]:
                    emission[train[i][j][1]][train[i][j][0]] += 1
                else:
                    emission[train[i][j][1]][train[i][j][0]] = 1
            else:
                emission[train[i][j][1]] = {}
                emission[train[i][j][1]][train[i][j][0]] = 1
                tags[train[i][j][1]] = 1

            if train[i][j][0] in words:
                words[train[i][j][0]] += 1
            else:
                words[train[i][j][0]] = 1

    for element in set2:
        for item in emission.keys():
            if element in emission[item]:
                emission[item][element] = math.log((emission[item][element] + 0.1)/(tags[item] + 0.1 * (len(set2) + 1)))
            else:
                emission[item][element] = math.log(0.1/(tags[item] + 0.1* (len(set2) + 1)))

    '''
    Trellis implementation
    '''
    final = []
    sum1 = sum(words.values())
    unknown = math.log(0.1/(sum1 + 0.1 * (len(set2) + 1)))
    states = list(emission.keys())
    for a in range(len(test)):
        prediction = []
        viterbi = np.empty((len(states), len(test[a])- 2))
        backpointer = np.empty((len(states), len(test[a])- 2))
        for i in range(len(states)):
                if test[a][1] in emission[states[i]]:
                    viterbi[i][0] = initial[states[i]] + emission[states[i]][test[a][1]]
                else:
                    viterbi[i][0] = initial[states[i]] + unknown
                backpointer[i][0] = -1
        if len(test[a]) == 3:
            prediction.append(("START", "START"))
            prediction.append((test[a][1], states[np.argmax(viterbi, axis = 0)[0]]))
            prediction.append(("END", "END"))
            final.append(prediction)
            continue
        for time_step in range(2, len(test[a])-1):
            for state in range(len(states)):
                probability = []
                for prev_state in range(len(states)-1):
                    if test[a][time_step] in emission[states[state]]:
                        probability.append((viterbi[prev_state][time_step-2]) + transition[states[prev_state]][states[state]] + emission[states[state]][test[a][time_step]])
                    else:
                        probability.append((viterbi[prev_state][time_step-2]) + transition[states[prev_state]][states[state]] + unknown)    
                backpointer[state][time_step-1] = np.argmax(probability)
                viterbi[state][time_step-1] = probability [int(backpointer[state][time_step-1])]
        best_path_pointer = backpointer[(np.argmax(viterbi, axis = 0))[-1]][-1]
        prediction.append(("END", "END"))
        prediction.append((test[a][-2], states[(np.argmax(viterbi, axis = 0))[-1]]))
        i = -3
        while best_path_pointer != -1 :
                prediction.append((test[a][i], states[int(best_path_pointer)]))
                best_path_pointer = backpointer[int(best_path_pointer)][i+1]
                i = i - 1
        prediction.append(("START", "START"))
        final.append(prediction[::-1])
    return final
