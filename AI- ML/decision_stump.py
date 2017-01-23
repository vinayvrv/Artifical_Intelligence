import time
import math
import random
import sys
train = sys.argv[1]
test = sys.argv[2]
algo = sys.argv[3]
stump_count  = int(sys.argv[4])
def read_file():
    global class_lab
    global image_data
    file = open('train-data.txt', 'r')
    for line in file:
        l = [int(x) for x in line.strip('\n').split()[1:]]
        image_data.append(l[1:])
        class_lab.append(l[0])
        # print len(l)
    # return image_data
def best_stumpfor_zero():
    global image_data
    global weights
    min_error = 10e10
    # error = 0
    # for i in range(len(image_data)):
    #     if image_data[i][col[0]] > image_data[i][col[1]]:
    #         if class_lab[i] != classify_label:
    #             error += D[i]
    # return [col[0], col[1], error]

    for i in range(len(image_data[0])):
        # print i
        for z in range(1,len(image_data[0])-i):
            error = 0
            total_weight = 0
            for j in range(len(image_data)):
                if image_data[j][i] > image_data[j][i+z]:
                    if class_lab[j] != classify_label:
                        error += D[j]
                        total_weight += D[j]

            # print error
            # error = error/total_weight
            run = 0
            if len(best_columns) > 0:
                for key in best_columns:
                    if best_columns[key][0] == i and best_columns[key][1] == i+z:
                        # print "Found it"
                        run = 1
            if min_error > error and run == 0:
                    min_error = error
                    best_col1 = i
                    best_col2 = i+z
    # print min_error
    return [best_col1, best_col2, min_error]

def create_ensemble():
    global D
    global image_data
    global Dt
    global class_lab
    global et
    global alpha
    global best_columns
    for i in range(stump_count):
        col = random.sample(xrange(0,191),2)
        # print i
        best_cols = best_stumpfor_zero()
        best_columns[i] = best_cols
        # print best_cols
        et[i] = best_cols[2]
        alpha[i] = math.log((1-et[i])/et[i])
        Dt[i] = D
        # print len(D), len(image_data), alpha, et
        for j in range(len(D)):
            # print "j", j, image_data[j][len(image_data[0])-1]
            if image_data[j][best_cols[0]] > image_data[j][best_cols[1]] and class_lab[j] != classify_label:
                D[j] = D[j]*math.exp(alpha[i])
            else:
                D[j] = D[j]*math.exp(-alpha[i])
        D = [x/sum(D) for x in D]
start_time = time.time()
classes = [0,90,180,270]
alpha1 = {}
columns = {}
for d in range(len(classes)):
    alpha1[classes[d]] = []
    columns[classes[d]] = []
class_lab = []
image_data = []
read_file()
for m in range(len(classes)):
    classify_label = classes[m]
    best_columns = {}
    Dt = {}
    et = {}
    alpha = {}
    weights = [1.0/len(class_lab) for x in range(len(class_lab))]
    D = [x/sum(weights) for x in weights]
    create_ensemble()
    f= open("test.txt","a")
    f.write(str(classify_label)+'\n')
    for key in best_columns:
        f.write(str(best_columns[key]) + '\n' + str(alpha[key]) + '\n' + str(et[key]) + '\n')
        alpha1[classify_label].append(alpha[key])
        columns[classify_label].append(best_columns[key])
    # print classify_label,best_columns, alpha, et

print("--- %s seconds ---" % (time.time() - start_time))
#test data

alpha = alpha1
class_lab = []
image_data1 = []
image_ids = []
def read_file1():
    global class_lab
    global image_data1
    file = open('test-data.txt', 'r')
    for line in file:
        p = [x for x in line.strip('\n').split()][0]
        l = [int(x) for x in line.strip('\n').split()[1:]]
        image_data1.append(l[1:])
        class_lab.append(l[0])
        image_ids.append(p)


read_file1()
sum = {}
outputs = []
for j in range(len(image_data1)):
    for key in alpha:
        sum[key] = 0
        for i in range(stump_count):
            if image_data1[j][columns[key][i][0]] > image_data1[i][columns[key][i][1]]:
                sum[key] += alpha[key][i]
            else:
                sum[key] -= alpha[key][i]

    max = -1
    for key in sum:
        if sum[key] > max:
            max = sum[key]
            output = key
    outputs.append(output)
# print outputs
# print class_lab
errors = 0
conf_matrix = {}
unique_classes = list(set(class_lab))
for j in range(len(unique_classes)):
    for k in range(len(unique_classes)):
        conf_matrix[(unique_classes[j], unique_classes[k])] = 0
for k in range(len(outputs)):
    key = (class_lab[k], outputs[k])
    if outputs[k] != class_lab[k]:
        errors += 1
        conf_matrix[key] += 1
    else:
        conf_matrix[key] += 1

print "Confusion Matrix"
print "     ",(str(unique_classes))[1:-1]
for key in range(len(unique_classes)):
    l = []
    for key1 in range(len(unique_classes)):
        l.append(conf_matrix[(unique_classes[key],unique_classes[key1])])
    print unique_classes[key],' ',l

f = open("adaboost_output.txt", 'w')
for t in range(len(outputs)):
    f.write((image_ids[t])+' '+str(outputs[t])+'\n')
# print errors
# print conf_matrix
print "Accuracy is", (1-float(errors)/len(class_lab))*100,"%"