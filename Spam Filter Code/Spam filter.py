# Code for question1 part a and part b

# write up is provided as separate file uploaded on github

#
#
import os
import random
import math
import string

import numpy as np
import math
from math import sqrt
from math import pow
import time
import sys



mode= sys.argv[1]
technique= sys.argv[2]
direc=sys.argv[3]
modelfile=sys.argv[4]

if mode.lower() == "train" and technique.lower()== "bayes":
    spam_dic={}
    nospam_dic={}
    spam_count=0
    nospam_count=0
    spam1={}
    nospam1={}

    for filename in os.listdir(direc+os.sep+"spam"):
        if filename!='cdms':
            new =open(direc+os.sep+"spam"+os.sep+filename,"r")
            indata=new.read().lower()
            indata=indata.translate(None,string.punctuation).split()
            new.close()
            spam_count+=1
            for i in indata:
                if i not in spam1:
                    spam1[i] = [spam_count]
                else:
                    lb=spam1[i]
                    if spam_count not in lb:
                        lb.append(spam_count)
                        spam1[i]=lb
                if i not in spam_dic:
                    spam_dic[i] = 1
                else:
                    spam_dic[i] += 1


    for filename in os.listdir(direc+os.sep+"notspam"):
        if filename!='cdms':
            new =open(direc+os.sep+"notspam"+os.sep+filename,"r")
            indata=new.read().lower()
            indata=indata.translate(None,string.punctuation).split()
            new.close()
            nospam_count += 1
            for i in indata:
                if i not in nospam1:
                    nospam1[i] = [nospam_count]
                else:
                     lb = nospam1[i]
                     if nospam_count not in lb:
                        lb.append(nospam_count)
                        nospam1[i] = lb
                if i not in nospam_dic:
                    nospam_dic[i] = 1
                else:
                    nospam_dic[i] += 1


    main_dic={}

    spam_prior= (spam_count*1.0/(spam_count +nospam_count))
    nospam_prior=(nospam_count*1.0/(spam_count + nospam_count))


    main_dic['spam']=spam_dic
    main_dic['nospam']=nospam_dic
    main_dic['spam1']=spam1
    main_dic['nospam1']=nospam1
    main_dic['spam_prior']=spam_prior
    main_dic['nospam_prior']=nospam_prior
    main_dic['spam_count']=spam_count
    main_dic['nospam_count']=nospam_count
    top10spam=[]
    top10spambinary = []
    base_spam = sum(main_dic['spam'].values())
    base_nospam = sum(main_dic['nospam'].values())
    for word in main_dic['spam']:
        pwgivenspam = math.log((main_dic['spam'].get(word, 0) + 1 * 1.0 / base_spam)*main_dic['spam_prior'])# adding soothing
        p_of_word=((main_dic['spam'].get(word, 0) + 1 * 1.0 / base_spam)*main_dic['spam_prior'])+((main_dic['nospam'].get(word, 0) + 1 * 1.0 / base_nospam)*main_dic['nospam_prior'])
        final_prob=pwgivenspam - math.log(p_of_word)
        top10spam.append((word,final_prob))
    top=sorted(top10spam,reverse=True,key=lambda x:x[1])

    for word in main_dic['spam1']:
        three = main_dic['spam1'].get(word, 0)
        if three == 0:
            pwgivenspam = (1.00 / main_dic['spam_count'])*main_dic['spam_prior']
        else:
            pwgivenspam = (len(main_dic['spam1'].get(word, 0))+1 * 1.0 / main_dic['spam_count'])*main_dic['spam_prior']


        four = main_dic['nospam1'].get(word, 0)
        if four == 0 :
            p_of_word =((1.00 / main_dic['nospam_count'])*main_dic['nospam_prior'])+ pwgivenspam
        else:
            p_of_word =((len(main_dic['nospam1'].get(word, 0)) +1* 1.0) / main_dic['nospam_count'])*main_dic['nospam_prior']
            p_of_word=p_of_word+pwgivenspam

        final_prob = math.log(pwgivenspam) - math.log(p_of_word)
        top10spambinary.append((word, final_prob))
    topbinary=sorted(top10spambinary,reverse=True, key=lambda x: x[1])

    print "***** Ten Top spam associated words for Frequency based are*****"
    for word in top[:10] :
        print word[0]
    print"\n"
    print "***** Ten Least Spam associated words for Frequency based are*****"
    for word in top[-10:] :
        print word[0]
    print"\n"

    print "***** Ten Top spam associated words for Binary based are*****"
    for word in topbinary[:10] :
        print word[0]
    print"\n"
    print "***** Ten Least Spam associated words for Binary based are*****"
    for word in topbinary[-10:] :
        print word[0]
    print"\n"


    stored_data=open(modelfile+".txt","w") # its unclear where to write the file hence writing it in the home directory
    stored_data.write(str(main_dic))
    stored_data.close()

if mode.lower() == "test" and technique.lower()== "bayes":

    # print "TESTING"
    newdic=open(modelfile+".txt","r")
    main_dic=eval(newdic.read())
    newdic.close()
    prediction=[]
    actual_label=[]
    prediction1=[]

    base_spam = sum(main_dic['spam'].values())
    base_nospam = sum(main_dic['nospam'].values())
    for filename in os.listdir(direc+os.sep+"spam"):
        if filename!='cdms':
            new =open(direc+os.sep+"spam"+os.sep+filename,"r")
            indata=new.read().lower()
            data=indata.translate(None,string.punctuation).split()
            new.close()
            actual_label.append((filename,1))
            non_spammer=0
            spammer=0
            spammer1=0
            non_spammer1=0
            for word in data:
                if len(word)>0:
                    spammer+= math.log(main_dic['spam'].get(word,0)+1*1.0/base_spam)
                    non_spammer += math.log(main_dic['nospam'].get(word, 0)+1*1.0/base_nospam)
                    three= main_dic['spam1'].get(word,0)
                    if three ==0:
                        spammer1+=math.log(1.00/main_dic['spam_count'])
                    else:
                        spammer1+=math.log(len(main_dic['spam1'].get(word, 0))  * 1.0 / main_dic['spam_count'])
                    four= main_dic['nospam1'].get(word,0)
                    if four ==0:
                        non_spammer1+=math.log(1.00/main_dic['nospam_count'])
                    else:
                        non_spammer1 +=  math.log(len(main_dic['nospam1'].get(word, 0))*1.0/ main_dic['nospam_count'])
            prob_spam=(spammer +math.log( main_dic['spam_prior']))
            prob_nospam =(non_spammer + math.log(main_dic['nospam_prior']))
            prob_spam1=(spammer1 +math.log(main_dic['spam_prior']))
            prob_nospam1 =(non_spammer1 + math.log(main_dic['nospam_prior']))

            if (prob_nospam > prob_spam):
                prediction.append((filename, 0))
            elif (prob_nospam <prob_spam):  # <=0.5:
                prediction.append((filename, 1))
            elif (prob_nospam ==prob_spam):
                prediction.append((filename, random.choice([1, 0])))
            if (prob_nospam1 > prob_spam1):
                prediction1.append((filename, 0))
            elif (prob_nospam1 <prob_spam1):  # <=0.5:
                prediction1.append((filename, 1))
            elif (prob_nospam1 ==prob_spam1):
                prediction1.append((filename, random.choice([1, 0])))

    for filename in os.listdir(direc+os.sep+"notspam"):
        if filename!='cdms':
            new =open(direc+os.sep+"notspam"+os.sep+filename,"r")
            indata=new.read().lower()
            data=indata.translate(None,string.punctuation).split()
            new.close()
            actual_label.append((filename,0))
            non_spammer=0
            spammer=0
            spammer1=0
            non_spammer1=0
            for word in data:
                if len(word)>0:
                    spammer += math.log(main_dic['spam'].get(word, 0) + 1 * 1.0 / base_spam)  # adding soothing
                    non_spammer += math.log(main_dic['nospam'].get(word, 0) + 1 * 1.0 / base_nospam)  # adding soothing
                    three = main_dic['spam1'].get(word, 0)
                    if three == 0:
                        spammer1 += math.log(1.00 / main_dic['spam_count'])
                    else:
                        spammer1 += math.log(len(main_dic['spam1'].get(word, 0))* 1.0 / main_dic['spam_count'])
                    four = main_dic['nospam1'].get(word, 0)
                    if four == 0:
                        non_spammer1 += math.log(1.00 / main_dic['nospam_count'])
                    else:
                        non_spammer1 += math.log(len(main_dic['nospam1'].get(word, 0)) * 1.0 / main_dic['nospam_count'])
            prob_spam=(spammer +math.log( main_dic['spam_prior']))#can optimize here
            prob_nospam =(non_spammer + math.log(main_dic['nospam_prior']))
            prob_spam1=(spammer1 +math.log(main_dic['spam_prior']))#can optimize here
            prob_nospam1 =(non_spammer1 + math.log(main_dic['nospam_prior']))

            if (prob_nospam > prob_spam):
                prediction.append((filename, 0))
            elif (prob_nospam <prob_spam):  # <=0.5:
                prediction.append((filename, 1))
            elif (prob_nospam ==prob_spam):
                prediction.append((filename, random.choice([1, 0])))
            if (prob_nospam1 > prob_spam1):
                prediction1.append((filename, 0))
            elif (prob_nospam1 <prob_spam1):  # <=0.5:
                prediction1.append((filename, 1))
            elif (prob_nospam1 ==prob_spam1):
                prediction1.append((filename, random.choice([1, 0])))

    def confusion_matrix(prediction,actual_label):
        true_pos=0
        true_neg=0
        false_pos=0
        false_neg=0
        for i in range (len(actual_label)):

            if actual_label[i][1]==1:
                if prediction[i][1]==actual_label[i][1]:
                    true_pos += 1
                else:
                    false_neg+=1

            elif actual_label[i][1]==0:
                if prediction[i][1] == actual_label[i][1]:
                    true_neg+=1
                else:
                    false_pos+=1
        print "                       Predicted Spam            Predicted Not Spam "
        print "Actual Spam                  "+str(true_pos)+"                    "+str(false_neg)
        print "Actual Not Spam              "+str(false_pos)+"                      " +str(true_neg)

    def accuracy(predict,actual):
        correct=0
        for i in range (len(actual)):
            if predict[i][1]==actual[i][1]:
                correct+=1
        return str(round((correct*1.0/len(actual)*1.0)*100,4))+" %"

    print  "Accuracy with Freq of occurrence",accuracy(prediction,actual_label)
    print
    print "                          Confusion Matrix with Freq"
    confusion_matrix(prediction,actual_label)
    print
    print "Accuracy with Binary appearance",accuracy(prediction1,actual_label)
    print
    print "                          Confusion Matrix Binary Appearance"
    confusion_matrix(prediction1,actual_label)
