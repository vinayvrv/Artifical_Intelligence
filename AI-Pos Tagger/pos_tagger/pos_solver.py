###################################
# CS B551 Fall 2016, Assignment #3
#
# Ritesh Agarwal(riteagar), Vinay Vernekar(vrvernek), Andrew Patterson(andnpatt)
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####
#Problem foundation
#We have created the transition function, emission function, function to calculate probability of a word given a label
#train function is to just calculate number of times a word occurs, pos occurs, transition pos, emission count
#For the second part we calculated the probability of word being a pos multiplied by probability of a pos
#Implemented viterbi for the third part
#For the last part when calculating probability for a tag (P(s_i|s_i-1, s_i-2, w_i-1, w_i-2)
# we tried two approaches first taking the max probability for each tag and passing it forward
#second summing over all probabilities for each tag and passing it forward, we get better results with the secong option
#There is not much difference in the accuracy of words but the accuracy of sentences increases by 3-4%

#How we calculated the probaility for a word given previous words and tags:
# we took the sum of the probability of previous two tags multiplied by the probability of both sequences occurring,
# multiplied by transition and emission probabilites and taking the max among all tags.

import random
import math
import copy

SMALL_CONSTANT = 10e-24

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    pos_total = 0
    c_word_pos = {}
    c_pos = {}
    c_pos_trans = {}
    c_pos_next_trans = {}

    def probPos(self, pos):
        global SMALL_CONSTANT
        if self.c_pos.get(pos, -1) != -1:
            return float(self.c_pos[pos]) / float(self.pos_total)
        else:
            return SMALL_CONSTANT

    def probNextPos(self, npos, pos):
        global SMALL_CONSTANT
        if self.c_pos_trans.get((npos, pos), -1) != -1:
            return float(self.c_pos_trans[npos, pos]) / float(self.c_pos[pos])
        else:
            return SMALL_CONSTANT

    def prob2Pos(self, npos, pos):
        global SMALL_CONSTANT
        if self.c_pos_next_trans.get((npos, pos), -1) != -1:
            return float(self.c_pos_next_trans[npos, pos]) / float(self.c_pos[pos])
        else:
            return SMALL_CONSTANT

    def probWordPos(self, word, pos):
        global SMALL_CONSTANT
        if self.c_word_pos.get((word, pos), -1) != -1:
            return float(self.c_word_pos[word, pos]) / float(self.c_pos[pos])
        else:
            return SMALL_CONSTANT


    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        sum = math.log(self.probPos(label[0])) + math.log(self.probWordPos(sentence[len(sentence)-1], label[len(label)-1]))
        for i in range(len(sentence)-1):
            sum += math.log(self.prob2Pos(label[i+1],label[i])) + math.log(self.probWordPos(sentence[i], label[i]))
        return sum

    # Do the training!
    #
    def train(self, data):
        # data[0] = sentence 0
        # data[0][0] = words
        # data[0][1] = pos
        for sentence in data:
            for i in range(len(sentence[0])):
                pos = sentence[1][i]
                w = sentence[0][i]
                # increment count of word | pos
                self.c_word_pos[w, pos] = self.c_word_pos.get((w, pos), 0) + 1
                # increment count of pos
                self.c_pos[pos] = self.c_pos.get(pos, 0) + 1
                self.pos_total = self.pos_total + 1
                # increment count of pos transition
                if i + 1 != len(sentence[0]):
                    next_pos = sentence[1][i+1]
                    self.c_pos_trans[next_pos, pos] = self.c_pos_trans.get((next_pos, pos), 0) + 1
                if i < len(sentence[0])-2:
                    next_next_pos = sentence[1][i+2]
                    self.c_pos_next_trans[next_next_pos, pos] = self.c_pos_next_trans.get((next_next_pos, pos), 0) + 1

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        parts = self.c_pos.keys()
        ans = []
        ans_prob = []
        for word in sentence:
            best = ('none', -1)
            for pos in parts:
                prob = self.probWordPos(word, pos) * self.probPos(pos)
                # print word + ': ' + `self.c_word_pos.get((word, pos), 'fail')` + ", " + `self.c_pos.get(pos, 'fail')`
                if prob > best[1]:
                    best = (pos, prob)
            ans.append(best[0])
            ans_prob.append(best[1])
        return [[ans], [ans_prob]]

    def hmm(self, sentence):
        ans = []
        ans_prob = []
        l = {}
        parts = self.c_pos.keys()
        max_prob = ('none', -1)
        #Calcuate the most probable pos for first word
        for ps in parts:
            probability = self.probWordPos(sentence[0], ps)
            l[ps] = probability
            if max_prob[1] < probability:
                max_prob = (ps, probability)
        ans.append(max_prob[0])
        ans_prob.append(max_prob[1])
        l_temp = {}
        #Applying Viterbi to the sequence
        for word in sentence[1:len(sentence)]:
            max_prob = ('none', -1)
            for pos in parts:
                l_temp[pos] = -1
                for prev_pos in parts:
                    prob = self.probWordPos(word, pos)*self.probNextPos(pos, prev_pos)*l[prev_pos]
                    #To store maximum probability for a pos
                    if l_temp[pos] < prob:
                        l_temp[pos] = prob
                    #To store maximum probability for a word among all pos
                    if max_prob[1] < prob:
                        max_prob = (pos, prob)
            l = copy.deepcopy(l_temp)
            ans.append(max_prob[0])
            ans_prob.append(max_prob[1])
        return [ [ans], [ans_prob] ]

    def complex(self, sentence):
        ans = []
        ans_prob = []
        l = {}
        l1 = {}
        parts = self.c_pos.keys()
        max_prob = ('none', 10e25)
        #Calcuate the most probable pos for first word by taking minimum of negative log
        for ps in parts:
            probability = -1*(math.log(self.probWordPos(sentence[0], ps),10))
            l[ps] = probability
            if max_prob[1] > probability:
                max_prob = (ps, probability)
        ans.append(max_prob[0])
        ans_prob.append(max_prob[1])
        max_prob = ('none', 10e25)
        if len(sentence) > 1:
            #For the second word there is only one predecessor so calculated separately
            for pos1 in parts:
                l1[pos1] = 0
                for prev_pos1 in parts:
                    prob1 = -1*(math.log(self.probWordPos(sentence[1], pos1),10)+math.log(self.probNextPos(pos1, prev_pos1),10)+math.log(l[prev_pos1],10))
                    if prob1 < 0:
                        prob1 = .000001
                    #calculate summation of log probability for a pos
                    l1[pos1] += prob1
            l_max = min(l1, key=lambda i: l1[i])
            max_prob = (l_max, l1[l_max])
            ans.append(max_prob[0])
            ans_prob.append(max_prob[1])
            l_temp = {}
            #Applying Viterbi like algorithm to the sequence
            for word in sentence[2:len(sentence)]:
                max_prob = ('none', 10e25)
                for pos in parts:
                    l_temp[pos] = 0
                    for prev_pos in parts:
                        for prev_prev_pos in parts:
                            prob = -1*(math.log(self.probWordPos(word, pos),10)+ math.log(self.probNextPos(pos, prev_pos),10) + math.log(l[prev_pos],10) + math.log(l1[prev_prev_pos],10) + math.log(self.prob2Pos(pos, prev_prev_pos),10))
                            if prob < 0:
                                prob = .000001
                            l_temp[pos] += prob
                l1_max = min(l_temp, key=lambda i: l_temp[i])
                max_prob = (l1_max, l_temp[l1_max])
                l = copy.deepcopy(l1)
                l1 = copy.deepcopy(l_temp)
                ans.append(max_prob[0])
                ans_prob.append(max_prob[1])
        return [ [ans], [ans_prob] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"
