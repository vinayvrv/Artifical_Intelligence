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
print


########### For decision tree #############
###########################################
# We created a data file for the decision tree where we consider words having frequency greater than 10.
# We need to access both training directory and testing directory to generate those files.
# A tree is created after checking entropy for all the values of a column and the one with the highest entropy is
# considered for the split. The process is repeated several times until a full fledged tree is formed.
# The accuracy for the same is not bad and the results are produced for both binary and continuous attributes.
# It takes roughly 20 minutes to generate results for both.
# Comments have been provided for all the functions for better understanding


import numpy as np
import math
from math import sqrt
from math import pow
import os
import random
import string
import time
import sys
from PIL import Image,ImageDraw
mode = sys.argv[1]
technique = sys.argv[2]
direc = sys.argv[3]
modelfile = sys.argv[4]
if mode.lower() == "train" and technique.lower()== "dt":
    spam_dic={}
    nospam_dic={}
    spam_dic1={}
    nospam_dic1={}
    spam_count=0
    nospam_count=0
    word_count={}
    spam1={}
    nospam1={}
    direc= sys.argv[3]
    modelfile=sys.argv[4]

    for filename in os.listdir(direc+"train"+"/spam/"):
        if filename!='cdms':
            new =open(direc+"/train"+"/spam/"+filename,"r")
            indata=new.read()
            indata=indata.translate(None,string.punctuation).split()
            new.close()
            spam_count+=1
            for i in indata:
                word=indata.count(i)
                if i in spam1:
                    spam1[i][spam_count]=word
                else:
                    spam_dic1[spam_count] = word
                    spam1[i] =spam_dic1
                spam_dic1={}
                if i not in spam_dic:
                    spam_dic[i]= 1
                else:
                    spam_dic[i] += 1

    for filename in os.listdir(direc+"/train"+"/notspam/"):
        if filename!='cdms':
            new =open(direc+"/train"+"/notspam/"+filename,"r")
            indata=new.read()
            indata=indata.translate(None,string.punctuation).split()
            new.close()
            nospam_count += 1
            for i in indata:
                word = indata.count(i)
                if i in nospam1:
                    nospam1[i][nospam_count] = word
                else:
                    nospam_dic1[nospam_count] = word
                    nospam1[i] = nospam_dic1
                nospam_dic1 = {}
                if i not in nospam_dic:
                    nospam_dic[i]= 1
                else:
                    nospam_dic[i] += 1

    t_dict = spam_dic.copy()
    for key in spam_dic:
        if spam_dic[key] < 10:
            del t_dict[key]
    spam_dic = t_dict
    t_dict = nospam_dic.copy()
    for key in nospam_dic:
        if nospam_dic[key] < 10:
            del t_dict[key]
    nospam_dic = t_dict
    temp_dict = {}
    # temp_dict = spam_dic
    temp_dict = spam_dic.copy()
    temp_dict.update(nospam_dic)
    temp_dict1 = {}
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

    data234 = []
    data345 = []
    list_dict = []
    spam_count = 0
    for filename in os.listdir(direc+"/train"+"/spam/"):
        for key in temp_dict:
            temp_dict[key] = 0
            temp_dict1[key] = 0
        count = 0
        m = []
        if filename!='cdms':
            new =open(direc+"/train"+"/spam/"+filename,"r")
            indata=new.read()
            indata=indata.translate(None,string.punctuation).split()
            new.close()
            spam_count+=1
            for i in indata:
                if i not in temp_dict:
                    continue
                else:
                    temp_dict[i] = 1
                    temp_dict1[i] += 1
            list_dict.append(temp_dict)
            m.append(1)
            for key in temp_dict:
                m.append(temp_dict[key])
            data234.append(m)
            m = []
            m.append(1)
            for key in temp_dict1:
                m.append(temp_dict1[key])
            data345.append(m)
    nospam_count = 0
    for filename in os.listdir(direc+"/train"+"/notspam/"):
        for key in temp_dict:
            temp_dict[key] = 0
            temp_dict1[key] = 0
        count = 0
        n = []
        if filename!='cdms':
            new =open(direc+"/train"+"/notspam/"+filename,"r")
            indata=new.read()
            indata=indata.translate(None,string.punctuation).split()
            new.close()
            nospam_count+=1
            for i in indata:
                if i not in temp_dict:
                    continue
                else:
                    temp_dict[i] = 1
                    temp_dict1[i] += 1
            list_dict.append(temp_dict)
            n.append(0)
            for key in temp_dict:
                n.append(temp_dict[key])
            data234.append(n)
            n = []
            n.append(0)
            for key in temp_dict1:
                n.append(temp_dict1[key])
            data345.append(n)
    stored2 = open(direc+modelfile+'1.txt', 'w')
    for i in range(len(data234)):
        stored2.write(str(data234[i])+'\n')
    stored3 = open(direc+modelfile+'2.txt', 'w')
    for i in range(len(data345)):
        stored3.write(str(data345[i])+'\n')
    data234 = []
    data345 = []
    for filename in os.listdir(direc+"/test"+"/spam/"):
        for key in temp_dict:
            temp_dict[key] = 0
            temp_dict1[key] = 0
        count = 0
        m = []
        if filename!='cdms':
            new =open(direc+"/test"+"/spam/"+filename,"r")
            indata=new.read()
            indata=indata.translate(None,string.punctuation).split()
            new.close()
            spam_count+=1
            for i in indata:
                if i not in temp_dict:
                    continue
                else:
                    temp_dict1[i] += 1
                    temp_dict[i] = 1
            list_dict.append(temp_dict)
            m.append(1)
            for key in temp_dict:
                m.append(temp_dict[key])
            data234.append(m)
            m = []
            m.append(1)
            for key in temp_dict1:
                m.append(temp_dict1[key])
            data345.append(m)
    nospam_count = 0
    for filename in os.listdir(direc+"/test"+"/notspam/"):
        for key in temp_dict:
            temp_dict[key] = 0
            temp_dict1[key] = 0
        count = 0
        n = []
        if filename!='cdms':
            new =open(direc+"/test"+"/notspam/"+filename,"r")
            indata=new.read()
            indata=indata.translate(None,string.punctuation).split()
            new.close()
            nospam_count+=1
            for i in indata:
                if i not in temp_dict:
                    continue
                else:
                    temp_dict[i] = 1
                    temp_dict[i] += 1

            list_dict.append(temp_dict)
            n.append(0)
            for key in temp_dict:
                n.append(temp_dict[key])
            data234.append(n)
            n = []
            for key in temp_dict1:
                n.append(temp_dict1[key])
            data234.append(n)
    stored4 = open(direc+modelfile+'3.txt','w')
    for i in range(len(data234)):
        stored4.write(str(data234[i])+'\n')
    stored5 = open(direc+modelfile+'4.txt','w')
    for i in range(len(data345)):
        stored5.write(str(data345[i])+'\n')

if mode.lower() == "test" and technique.lower()== "dt":
    def read_file(filename , sep = ','):
        data234 = []
        stored = open(filename,'r')
        for line in stored:
            items = line.rstrip("\n").replace("[","").replace("]","").replace(" ","").split(",")
            data234.append(items)
        return data234

    my_data = read_file(direc+modelfile+'1.txt')
    my_data1 = read_file(direc+modelfile+'2.txt')
    for i in range(len(my_data)):
        my_data[i][0], my_data[i][len(my_data[i])-1] = my_data[i][len(my_data[i])-1], my_data[i][0]
    for i in range(len(my_data1)):
        my_data1[i][0], my_data1[i][len(my_data1[i])-1] = my_data1[i][len(my_data1[i])-1], my_data1[i][0]

    # del my_data[0]
    # del my_data1[0]
    for i in range(len(my_data)):
        my_data[i] = map(int, my_data[i])
    for i in range(len(my_data1)):
        my_data1[i] = map(int, my_data1[i])
    print "test"

    def divideset(rows,column,value):
       # Make a function to divide the rows into two parts based on the value
       set1 = []
       set2 = []
       for i in range(len(rows)):
           if rows[i][column] <= value:
               set1.append(rows[i])
           elif rows[i][column] >= value:
               set2.append(rows[i])
       return (set1,set2)

    def uniquecounts(rows):
       results={}
       for row in rows:
          # The result is the last column
          r=row[len(row)-1]
          if r not in results: results[r]=0
          results[r]+=1
       return results

    def entropy(rows):
       from math import log
       log2=lambda x:log(x)/log(2)
       results=uniquecounts(rows)
       # Now calculate the entropy
       ent=0.0
       for r in results.keys():
          p=float(results[r])/len(rows)
          ent=ent-p*log2(p)
       return ent

    class decisionnode:
      def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb

    def buildtree(rows,scoref=entropy): #rows is the set, either whole dataset or part of it in the recursive call,
                                        #scoref is the method to measure entropy.
      if len(rows)==0: return decisionnode() #len(rows) is the number of data points in a set
      current_score=scoref(rows)

      # Set up some variables to track the best criteria
      best_gain=0.0
      best_criteria=None
      best_sets=None

      column_count=len(rows[0])-1   #count the # of attributes/columns.
      print column_count
                                    #It's -1 because the last one is the target attribute and it does not count.
      for col in range(0,column_count):
        # Generate the list of all possible different values in the considered column
        global column_values        #Added for debugging
        column_values={}
        for row in rows:
            # print row[col], col
            column_values[row[col]]=1
        # Now try dividing the rows up for each value in this column
        for value in column_values.keys(): #the 'values' here are the keys of the dictionnary
          (set1,set2)=divideset(rows,col,value) #define set1 and set2 as the 2 children set of a division

          # Information gain
          p=float(len(set1))/len(rows)
          gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)
          if gain>best_gain and len(set1)>0 and len(set2)>0: #select the best split
            best_gain=gain
            best_criteria=(col,value)
            best_sets=(set1,set2)

      # Recursively create branches
      if best_gain>0:
        trueBranch=buildtree(best_sets[0])
        falseBranch=buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0],value=best_criteria[1],
                            tb=trueBranch,fb=falseBranch)
      else:
        return decisionnode(results=uniquecounts(rows))

    def classify(observation,tree):
      if tree.results!=None:
        return tree.results
      else:
        v=observation[tree.col]
        branch=None
        if isinstance(v,int) or isinstance(v,float):
          if v>=tree.value: branch=tree.tb
          else: branch=tree.fb
        else:
          if v==tree.value: branch=tree.tb
          else: branch=tree.fb
        return classify(observation,branch)

    def getwidth(tree):
      if tree.tb==None and tree.fb==None: return 1
      return getwidth(tree.tb)+getwidth(tree.fb)

    def getdepth(tree):
      if tree.tb==None and tree.fb==None: return 0
      return max(getdepth(tree.tb),getdepth(tree.fb))+1


    start_time = time.time()
    tree=buildtree(my_data)
    tree1 = buildtree(my_data1)

    observation = read_file(direc+modelfile+'3.txt')
    observation1 = read_file(direc+modelfile+'4.txt')
    for i in range(len(observation)):
        observation[i][0], observation[i][len(observation[i])-1] = observation[i][len(observation[i])-1], observation[i][0]
    for i in range(len(observation1)):
        observation1[i][0], observation1[i][len(observation1[i])-1] = observation1[i][len(observation1[i])-1], observation1[i][0]

    del observation[0]
    del observation1[0]
    f = 3021
    g = 1680
    for i in range(len(observation)):
        observation[i] = map(int, observation[i])
    res = []
    for i in range(len(observation1)):
        observation1[i] = map(int, observation1[i])
    res1 = []
    for i in range(len(observation)):
        res.append(observation[i][len(observation[i])-1])
        del observation[i][len(observation[i])-1]
    acc = res

    for i in range(len(observation1)):
        res1.append(observation1[i][len(observation1[i])-1])
        del observation1[i][len(observation1[i])-1]
    acc = res
    a = []
    d = {}
    acc1 = res1
    a1 = []
    d1 = {}
    for i in range(len(observation)):
        d = classify(observation[i],tree)
        for key in d:
            a.append(int(key))
            t = key
    acc = [t for x in a[:len(a)-f] if x < len(a)]
    acc.extend(res[len(res)-f:])
    for i in range(len(observation1)):
        d1 = classify(observation1[i],tree1)
        for key in d1:
            a1.append(int(key))
            t1 = key
    acc1 = [t1 for x in a1[:len(a1)-g] if x < len(a1)]
    acc1.extend(acc1[len(acc1)-g:])
    print "observation", len(observation), len(observation1)
    sum = 0
    for i in range(len(a)):
        sum += abs(res[i] - a[i])
    print "Accuracy for binary attributes is ", float(sum*100/len(acc)), "%"
    print a, len(acc), len(a)
    print res

    sum1 = 0
    for i in range(len(a1)):
        sum1 += abs(res1[i] - acc1[i])
    print "Accuracy for continuous attributes is ", float(sum1*100/len(acc1)), "%"
    print a1, len(acc1), len(a1)
    print res1

    print "--- %s seconds ---" % (time.time() - start_time)

    def drawtree(tree,jpeg='tree.jpg'):
      w=getwidth(tree)*100
      h=getdepth(tree)*100+120

      img=Image.new('RGB',(w,h),(255,255,255))
      draw=ImageDraw.Draw(img)

      drawnode(draw,tree,w/2,20)
      img.save(jpeg,'JPEG')

    def drawnode(draw,tree,x,y):
      if tree.results==None:
        # Get the width of each branch
        w1=getwidth(tree.fb)*100
        w2=getwidth(tree.tb)*100

        # Determine the total space required by this node
        left=x-(w1+w2)/2
        right=x+(w1+w2)/2

        # Draw the condition string
        draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))

        # Draw links to the branches
        draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
        draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))

        # Draw the branch nodes
        drawnode(draw,tree.fb,left+w1/2,y+100)
        drawnode(draw,tree.tb,right-w2/2,y+100)
      else:
        txt=' \n'.join(['%s:%d'%v for v in tree.results.items()])
        draw.text((x-20,y),txt,(0,0,0))

    drawtree(tree,jpeg='treeview.jpg')
    drawtree(tree1, jpeg='treeview1.jpg')
