from __future__ import division
import numpy as np
from scipy import optimize
import time
import sys
train = sys.argv[1]
test = sys.argv[2]
algo = sys.argv[3]
nodes = 25
class Neural_net(object):
    def __init__(self, lamda=0, eps=0.12, nodes_num=nodes, maxiter=50):
        self.lamda = lamda
        self.eps = eps
        self.nodes_num = nodes_num
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid1
        self.maxiter = maxiter
        
    def square_sum(self, x):
        return np.sum(x ** 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def random_eps(self, x, y):
        return np.random.rand(y, x + 1) * 2 * self.eps - self.eps

    def combine_weights(self, weight_1, weight_2):
        return np.concatenate((weight_1.reshape(-1), weight_2.reshape(-1)))

    def uncombine_weights(self, weights, input_layer_size, nodes_num, num_labels):
        weight_1_start = 0
        weight_1_end = nodes_num * (input_layer_size + 1)
        weight_1 = weights[weight_1_start:weight_1_end].reshape((nodes_num, input_layer_size + 1))
        weight_2 = weights[weight_1_end:].reshape((num_labels, nodes_num + 1))
        return weight_1, weight_2
    
    def sigmoid1(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

#This function is used for gradient descent
    def calc_gradient(self, weights, input_layer_size, nodes_num, num_labels, X, y, lamda):
        weight_1, weight_2 = self.uncombine_weights(weights, input_layer_size, nodes_num, num_labels)
        global counweight_1
        m = X.shape[0]
        weight_1f = weight_1[:, 1:]
        weight_2f = weight_2[:, 1:]
        #Convert the labels into a matrix giving 1 to a label in a row
        Y = np.eye(num_labels)[y]

        delta1, delta2 = 0, 0
        for i, row in enumerate(X):
            a1, z2, a2, z3, a3 = self._output_label(row, weight_1, weight_2)
            # Backpropagation is applied here
            d3 = a3 - Y[i, :].T
            d2 = np.dot(weight_2f.T, d3) * self.activation_func_prime(z2)

            delta2 += np.dot(d3[np.newaxis].T, a2[np.newaxis])
            delta1 += np.dot(d2[np.newaxis].T, a1[np.newaxis])

        weight1_gradient = (1 / m) * delta1
        weight2_gradient = (1 / m) * delta2

        if lamda != 0:
            weight1_gradient[:, 1:] = weight1_gradient[:, 1:] + (lamda / m) * weight_1f
            weight2_gradient[:, 1:] = weight2_gradient[:, 1:] + (lamda / m) * weight_2f
        counweight_1 += 1
        return self.combine_weights(weight1_gradient, weight2_gradient)

#this function is used to claculate the data points in the final layer based on the weights calculated
    def _output_label(self, X, weight_1, weight_2):
        m = X.shape[0]
        ones = None
        if len(X.shape) == 1:
            ones = np.array(1).reshape(1,)
        else:
            ones = np.ones(m).reshape(m,1)
        # Input layer for the network
        a1 = np.hstack((ones, X))
        #One hidden Layer
        z2 = np.dot(weight_1, a1.T)
        a2 = self.activation_func(z2)
        a2 = np.hstack((ones, a2.T))
        # Final output layer
        z3 = np.dot(weight_2, a2.T)
        a3 = self.activation_func(z3)
        return a1, z2, a2, z3, a3

#This function is used for calculating cross entropy
    def calc_error(self, weights, input_layer_size, nodes_num, num_labels, X, y, lamda):
        global count
        weight_1, weight_2 = self.uncombine_weights(weights, input_layer_size, nodes_num, num_labels)
        # print y
        m = X.shape[0]
        Y = np.eye(num_labels)[y]

        q, w, e, r, h = self._output_label(X, weight_1, weight_2)
        pos_cost = -Y * np.log(h).T
        cost_neg = (1 - Y) * np.log(1 - h).T
        cost = pos_cost - cost_neg
        J = np.sum(cost) / m #Cross Entropy
        #If regularization is applied then we change the cost accordingly
        if lamda != 0:
            weight_1f = weight_1[:, 1:]
            weight_2f = weight_2[:, 1:]
            reg = (self.lamda / (2 * m)) * (self.square_sum(weight_1f) + self.square_sum(weight_2f))
            J = J + reg
        count += 1
        return J


    def train(self, X, y):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))

        w_1 = self.random_eps(input_layer_size, self.nodes_num)
        w_2 = self.random_eps(self.nodes_num, num_labels)
        weights0 = self.combine_weights(w_1, w_2)

        options = {'maxiter': self.maxiter}
        _res = optimize.minimize(self.calc_error, weights0, jac=self.calc_gradient, method="TNC",
                                 args=(input_layer_size, self.nodes_num, num_labels, X, y, 0), options=options)

        self.weight_1, self.weight_2 = self.uncombine_weights(_res.x, input_layer_size, self.nodes_num, num_labels)

    def predict(self, X):
        return self.predict_f(X).argmax(0)

    def predict_f(self, X):
        q, w, e, r, h = self._output_label(X, self.weight_1, self.weight_2)
        return h

count = 0
counweight_1 = 0
file = open(train)
data= file.readlines()

xtrain=[]
ytrain=[]
ytrain_id=[]

for i in data:
    i=i.strip()
    j=i.split()
    ytrain_id.append(j[0]) # photo id
    ytrain.append(int(j[1]))# label
    row = []
    for k in j[2:]:
        if k !="\n":
            row.append(k)
    xtrain.append(row)# features
X_train=np.array(xtrain)
X_train=X_train.astype(int)
y_train=np.array(ytrain)
y_train_id=np.array(ytrain_id)
X_train = X_train[:20000]
y_train = y_train[:20000]
xtest=[]
ytest=[]
xtest_id=[]

file = open(test)
data= file.readlines()
image_ids = []
for i in data:
    i = i.strip()
    j=i.split()
    image_ids.append(j[0])# photo id
    ytest.append(int(j[1]))# label
    row=[]
    for k in j[2:]:
        if k !='\n':
            row.append(k)
    xtest.append(row)# features

X_test=np.array(xtest)
X_test=X_test.astype(int)
y_test=np.array(ytest)
X_test_id=np.array(xtest_id)
y_train[y_train==180] = 2
y_train[y_train==90] = 1
y_train[y_train==270] = 3

start_time = time.time()
nn = Neural_net()
nn.train(X_train, y_train)
# print nn
y_predict = nn.predict(X_test)
y_predict[y_predict==2] = 180
y_predict[y_predict==1] = 90
y_predict[y_predict==3] = 270
errors = 0
conf_matrix = {}
unique_classes = list(set(y_test))
for j in range(len(unique_classes)):
    for k in range(len(unique_classes)):
        conf_matrix[(unique_classes[j], unique_classes[k])] = 0
for k in range(len(y_test)):
    key = (y_test[k], y_predict[k])
    if y_test[k] != y_predict[k]:
        key = (y_test[k], y_predict[k])
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

print "Accuracy is" , 1-(errors)/len(y_test)
print count, counweight_1
f = open("nnet_output.txt", 'w')
for t in range(len(y_predict)):
    f.write((image_ids[t])+' '+str(y_predict[t])+'\n')
print("--- %s seconds ---" % (time.time() - start_time))