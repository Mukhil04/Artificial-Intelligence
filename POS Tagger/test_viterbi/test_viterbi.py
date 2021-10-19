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
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math
import numpy as np

def main():
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    prediction = []
    
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""
    states = list(emission.keys())
    viterbi = np.empty((len(states), len(test[0])))
    backpointer = np.empty((len(states), len(test[0])))
    for i in range(len(states)):
        viterbi[i][0] = initial[states[i]] * emission[states[i]][test[0][0]]
        backpointer[i][0] = -1
    for time_step in range(1, len(test[0])):
        for state in range(len(states)):
            probability = []
            for prev_state in range(len(states)):
                probability.append((viterbi[prev_state][time_step-1]) * transition[states[prev_state]][states[state]] * emission[states[state]][test[0][time_step]])
            backpointer[state][time_step] = np.argmax(probability)
            viterbi[state][time_step] = probability [int(backpointer[state][time_step])]
    best_path_pointer = backpointer[(np.argmax(viterbi, axis = 0))[-1]][-1]
    prediction.append((test[0][-1], states[(np.argmax(viterbi, axis = 0))[-1]]))
    i = -2
    while best_path_pointer != -1 :
        prediction.append((test[0][i], states[int(best_path_pointer)]))
        best_path_pointer = backpointer[int(best_path_pointer)][i]
        i = i - 1
    prediction.reverse()
    print('Your Output is:',prediction,'\n Expected Output is:',output)
    print(emission)

if __name__=="__main__":
    main()