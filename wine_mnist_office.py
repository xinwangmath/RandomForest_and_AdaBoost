# Author: Xin Wang
# Email: xinwangmath@gmail.com

import numpy as np
import scipy as sp
import scipy.linalg
from sklearn import tree
import mllib as ml
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

# 1. load in data
code_path = os.getcwd()
head, tail = os.path.split(code_path)
wine_path = head + '/wine'
mnist_path = head + '/MNIST'
office_path = head + '/office'

wine_raw = pd.read_csv(wine_path + '/wine.data', header = None)
mnist_raw_train = pd.read_csv(mnist_path + '/train.csv', header = None).transpose()
mnist_raw_test = pd.read_csv(mnist_path + '/test.csv', header = None).transpose()

# office data load in, append to the office data analysis part
office_raw_train = pd.read_csv(office_path + '/train.csv', header = None)
office_raw_test = pd.read_csv(office_path + '/test.csv', header = None)


#print wine_raw.head(5)
#print mnist_raw_train.head(5)
#print mnist_raw_test.tail(5)

# 2. data preprocessing
#  for wine dataset, split into train and test, separate X and y
wine_train_index, wine_test_index = ml.split_dataset_index(wine_raw.shape[0],
    training_ratio = 0.75)
wine_X_train = (wine_raw.ix[wine_train_index]).as_matrix(columns = range(1, 14))
wine_y_train = wine_raw.ix[wine_train_index]
wine_y_train = wine_y_train[0].as_matrix()
wine_X_test = (wine_raw.ix[wine_test_index]).as_matrix(columns = range(1, 14))
wine_y_test = wine_raw.ix[wine_test_index]
wine_y_test = wine_y_test[0].as_matrix()

# for the mnist dataset, cast the class label type into int, separate X and y
mnist_raw_train[784] = mnist_raw_train[784].astype(int)
mnist_raw_test[784] = mnist_raw_test[784].astype(int)

mnist_X_train = mnist_raw_train.as_matrix(columns = range(784))
mnist_y_train = mnist_raw_train[784].as_matrix()
mnist_X_test  = mnist_raw_test.as_matrix(columns = range(784))
mnist_y_test  = mnist_raw_test[784].as_matrix()

# for the office dataset
# append to office data analysis part
office_n_cols = office_raw_test.shape[1]
office_raw_train[office_n_cols - 1] = office_raw_train[office_n_cols - 1].astype(int)
office_raw_test[office_n_cols - 1] = office_raw_test[office_n_cols -1].astype(int)

office_X_train = office_raw_train.as_matrix(columns = range(office_n_cols-3))
office_y_train = office_raw_train[office_n_cols-1].as_matrix()
office_X_test  = office_raw_test.as_matrix(columns = range(office_n_cols-3))
office_y_test  = office_raw_test[office_n_cols-1].as_matrix()



# 3. Data Analysis
# 3.1.1. RF for wine
# 3.1.1.1. max depth = None
reload(ml)
MAX = 1000
SUBS = 200
rf_wine_full = ml.RandomForestClassifier(n_trees = MAX, max_depth = None,
    random_state = 5)
rf_wine_full = rf_wine_full.fit(wine_X_train, wine_y_train)
# rf_wine_full_oobe = rf_wine_full.get_oobe_list()
train_errors = np.ones(SUBS)
test_errors = np.ones(SUBS)
oob_errors = np.ones(SUBS)

for i in xrange(1, SUBS+1):
    sub_rf = rf_wine_full.sub_randomForest(i * MAX//SUBS)
    train_errors[i-1] = sub_rf.compute_error(wine_X_train, wine_y_train)
    test_errors[i-1] = sub_rf.compute_error(wine_X_test, wine_y_test)
    oob_errors[i-1] = sub_rf.oobe()

num_of_trees = np.arange(1, SUBS + 1) * (MAX//SUBS)

plt.plot(num_of_trees, train_errors, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, oob_errors, 'b*-', linewidth = 1.5, label = 'out-of-bag error')
plt.plot(num_of_trees, test_errors, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend()
plt.title('wine, random forest error rate, trees grow to full extent', fontsize = 16)
plt.show()

rf_wine_5 = ml.RandomForestClassifier(n_trees = MAX, max_depth = 5,
    random_state = 5)
rf_wine_5 = rf_wine_5.fit(wine_X_train, wine_y_train)

for i in xrange(1, SUBS+1):
    sub_rf = rf_wine_5.sub_randomForest(i * MAX//SUBS)
    train_errors[i-1] = sub_rf.compute_error(wine_X_train, wine_y_train)
    test_errors[i-1] = sub_rf.compute_error(wine_X_test, wine_y_test)
    oob_errors[i-1] = sub_rf.oobe()

plt.plot(num_of_trees, train_errors, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, oob_errors, 'b*-', linewidth = 1.5, label = 'out-of-bag error')
plt.plot(num_of_trees, test_errors, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend()
plt.title('wine, random forest error rate, trees max depth = 5', fontsize = 16)
plt.show()

# generate confusion_matrix
sub_rf = rf_wine_full.sub_randomForest(2)
sub_rf_cm = confusion_matrix(wine_y_test, sub_rf.predict(wine_X_test))
rf_wine_full_cm = confusion_matrix(wine_y_test, rf_wine_full.predict(wine_X_test))
print sub_rf_cm
print rf_wine_full_cm

sub_rf = rf_wine_5.sub_randomForest(2)
sub_rf_cm_5 = confusion_matrix(wine_y_test, sub_rf.predict(wine_X_test))
rf_wine_5_cm = confusion_matrix(wine_y_test, rf_wine_5.predict(wine_X_test))
print sub_rf_cm_5
print rf_wine_5_cm

# 3.1.2. rf for mnist
MAX = 500
SUBS = 20
rf_mnist_full = ml.RandomForestClassifier(n_trees = MAX, max_depth = None,
    random_state = 5)
rf_mnist_full = rf_mnist_full.fit(mnist_X_train, mnist_y_train)

#rf_mnist_oobe = rf_mnist_full.get_oobe_list()
#plt.plot(rf_mnist_oobe)

train_errors = np.ones(SUBS)
test_errors = np.ones(SUBS)
oob_errors = np.ones(SUBS)

for i in xrange(1, SUBS+1):
    sub_rf = rf_mnist_full.sub_randomForest(i * MAX//SUBS)
    train_errors[i-1] = sub_rf.compute_error(mnist_X_train, mnist_y_train)
    test_errors[i-1] = sub_rf.compute_error(mnist_X_test, mnist_y_test)
    oob_errors[i-1] = sub_rf.oobe()

num_of_trees = np.arange(1, SUBS + 1) * (MAX//SUBS)

plt.plot(num_of_trees, train_errors, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, oob_errors, 'b*-', linewidth = 1.5, label = 'out-of-bag error')
plt.plot(num_of_trees, test_errors, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('MNIST, random forest error rate, trees grow to full extent', fontsize = 16)
plt.show()

rf_mnist_5 = ml.RandomForestClassifier(n_trees = MAX, max_depth = 5,
    random_state = 5)
rf_mnist_5 = rf_mnist_5.fit(mnist_X_train, mnist_y_train)

for i in xrange(1, SUBS+1):
    sub_rf = rf_mnist_5.sub_randomForest(i * MAX//SUBS)
    train_errors[i-1] = sub_rf.compute_error(mnist_X_train, mnist_y_train)
    test_errors[i-1] = sub_rf.compute_error(mnist_X_test, mnist_y_test)
    oob_errors[i-1] = sub_rf.oobe()

plt.plot(num_of_trees, train_errors, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, oob_errors, 'b*-', linewidth = 1.5, label = 'out-of-bag error')
plt.plot(num_of_trees, test_errors, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('MNIST, random forest error rate, trees max depth = 5', fontsize = 16)
plt.show()

# generate confusion_matrix
sub_rf = rf_mnist_full.sub_randomForest(2)
sub_rf_cm = confusion_matrix(mnist_y_test, sub_rf.predict(mnist_X_test))
rf_mnist_full_cm = confusion_matrix(mnist_y_test, rf_mnist_full.predict(mnist_X_test))
print sub_rf_cm
print rf_mnist_full_cm

sub_rf = rf_mnist_5.sub_randomForest(2)
sub_rf_cm_5 = confusion_matrix(mnist_y_test, sub_rf.predict(mnist_X_test))
rf_mnist_5_cm = confusion_matrix(mnist_y_test, rf_mnist_5.predict(mnist_X_test))
print sub_rf_cm_5
print rf_mnist_5_cm


# 3.3.1. adaboost for mnist
# 3.3.1.1. compute the error rates
reload(ml)
MAX = 1000
SUBS = 50
boost_mnist_1 = ml.AdaBoostClassifier(n_estimators = MAX, max_depth = 1, random_state = 5)
boost_mnist_1 = boost_mnist_1.fit(mnist_X_train, mnist_y_train)

mb_train_errors = np.ones(SUBS)
mb_test_errors = np.ones(SUBS)

for i in xrange(1, SUBS+1):
    sub_ada = boost_mnist_1.sub_adaboost(i * MAX//SUBS)
    mb_train_errors[i-1] = sub_ada.compute_error(mnist_X_train, mnist_y_train)
    mb_test_errors[i-1] = sub_ada.compute_error(mnist_X_test, mnist_y_test)

num_of_trees = np.arange(1, SUBS + 1) * (MAX//SUBS)

plt.plot(num_of_trees, mb_train_errors, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, mb_test_errors, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('MNIST, AdaBoost error rate, decision stump', fontsize = 16)
plt.show()


# max_depth = 10
boost_mnist_10 = ml.AdaBoostClassifier(n_estimators = MAX, max_depth = 10, random_state = 5)
boost_mnist_10 = boost_mnist_10.fit(mnist_X_train, mnist_y_train)

mb_train_errors = np.ones(SUBS)
mb_test_errors = np.ones(SUBS)

for i in xrange(1, SUBS+1):
    sub_ada = boost_mnist_10.sub_adaboost(i * MAX//SUBS)
    mb_train_errors[i-1] = sub_ada.compute_error(mnist_X_train, mnist_y_train)
    mb_test_errors[i-1] = sub_ada.compute_error(mnist_X_test, mnist_y_test)

num_of_trees = np.arange(1, SUBS + 1) * (MAX//SUBS)

plt.plot(num_of_trees, mb_train_errors, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, mb_test_errors, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('MNIST, AdaBoost error rate, max_depth = 10', fontsize = 16)
plt.show()

# 3.3.1.2. generate the confusion_matrix
sub_ada = boost_mnist_1.sub_adaboost(MAX//SUBS)
sub_ada_cm = confusion_matrix(mnist_y_test, sub_ada.predict(mnist_X_test))
ada_mnist_1_cm = confusion_matrix(mnist_y_test, boost_mnist_1.predict(mnist_X_test))
print sub_ada_cm
print ada_mnist_1_cm

sub_ada = boost_mnist_10.sub_adaboost(MAX//SUBS)
sub_ada_cm = confusion_matrix(mnist_y_test, sub_ada.predict(mnist_X_test))
ada_mnist_10_cm = confusion_matrix(mnist_y_test, boost_mnist_10.predict(mnist_X_test))
print sub_ada_cm
print ada_mnist_10_cm

# 3.3.2. adaboost for office dataset
# append the data loading and preprocessing codes here
MAX = 1000
SUBS = 50

# fit the models
# decision stump
boost_office_1 = ml.AdaBoostClassifier(n_estimators = MAX, max_depth = 1, random_state = 5)
boost_office_1 = boost_office_1.fit(office_X_train, office_y_train)
# max depth = 10
boost_office_10 = ml.AdaBoostClassifier(n_estimators = MAX, max_depth = 10, random_state = 5)
boost_office_10 = boost_office_10.fit(office_X_train, office_y_train)

# compute and plot the errors
ob_train_errors_1 = np.ones(SUBS)
ob_test_errors_1 = np.ones(SUBS)

for i in xrange(1, SUBS+1):
    sub_ada = boost_office_1.sub_adaboost(i * MAX//SUBS)
    ob_train_errors_1[i-1] = sub_ada.compute_error(office_X_train, office_y_train)
    ob_test_errors_1[i-1] = sub_ada.compute_error(office_X_test, office_y_test)

num_of_trees = np.arange(1, SUBS + 1) * (MAX//SUBS)

plt.plot(num_of_trees, ob_train_errors_1, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, ob_test_errors_1, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('office, AdaBoost error rate, decision stump', fontsize = 16)
plt.show()

ob_train_errors_10 = np.ones(SUBS)
ob_test_errors_10 = np.ones(SUBS)

for i in xrange(1, SUBS+1):
    sub_ada = boost_office_10.sub_adaboost(i * MAX//SUBS)
    ob_train_errors_10[i-1] = sub_ada.compute_error(office_X_train, office_y_train)
    ob_test_errors_10[i-1] = sub_ada.compute_error(office_X_test, office_y_test)

#num_of_trees = np.arange(1, SUBS + 1) * (MAX//SUBS)

plt.plot(num_of_trees, ob_train_errors_10, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, ob_test_errors_10, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('office, AdaBoost error rate, max depth = 10', fontsize = 16)
plt.show()

# generate the confusion_matrix
sub_ada = boost_office_1.sub_adaboost(MAX//SUBS)
sub_ada_cm_1= confusion_matrix(office_y_test, sub_ada.predict(office_X_test))
ada_office_1_cm = confusion_matrix(office_y_test, boost_office_1.predict(office_X_test))
#print sub_ada_cm_1
#print ada_office_1_cm

sub_ada = boost_office_10.sub_adaboost(MAX//SUBS)
sub_ada_cm_10= confusion_matrix(office_y_test, sub_ada.predict(office_X_test))
ada_office_10_cm = confusion_matrix(office_y_test, boost_office_10.predict(office_X_test))
#print sub_ada_cm_10
#print ada_office_10_cm

# plot the confusion matrix
office_labels = np.unique(office_y_test)
ml.plot_cm(sub_ada_cm_1, labels = office_labels,
    title = 'confusion matrix, AdaBoost with decision stump, 20 trees')
ml.plot_cm(ada_office_1_cm, labels = office_labels,
    title = 'confusion matrix, AdaBoost with decision stump, 1000 trees')
ml.plot_cm(sub_ada_cm_10, labels = office_labels,
    title = 'confusion matrix, AdaBoost with trees of depth 10, 20 trees')
ml.plot_cm(ada_office_10_cm, labels = office_labels,
    title = 'confusion matrix, AdaBoost with trees of depth 10, 1000 trees')

# 4.1. Dimension reduction with random forest and AdaBoost
# 4.1.1. mnist
#     choose a small sub training set, fit with random forest, compute feature
#         importances, then only use the most important features to do
#         classification.
sub_sample_size = mnist_X_test.shape[0]//10
mnist_X_sub, mnist_y_sub, sub_index = ml.resample(mnist_X_train, mnist_y_train,
    n_samples = sub_sample_size, replace = False, random_state = 5)
rf_mnist_sub = ml.RandomForestClassifier(n_trees = 200,
                                         max_depth = 10,
                                         random_state = 5)
rf_mnist_sub = rf_mnist_sub.fit(mnist_X_sub, mnist_y_sub)
mnist_feature_importance = rf_mnist_sub.get_feature_importances()
mnist_important_features = mnist_feature_importance.argsort()[-800:][::-1]
plt.plot(mnist_feature_importance[mnist_important_features], 'b.-', linewidth = 1.5)
plt.ylabel('feature importance', fontsize = 16)
plt.title('feature importance for mnist', fontsize = 16)
plt.show()

mnist_important_features = mnist_important_features[:200]

MAX = 300
SUB = 30
rf_mnist_important = ml.RandomForestClassifier(n_trees = MAX,
                                               max_depth = 5,
                                               random_state = 5)
rf_mnist_important = rf_mnist_important.fit(mnist_X_train[:, mnist_important_features], mnist_y_train)

important_train_error = np.zeros(SUB)
important_test_error = np.zeros(SUB)
important_oob_error = np.zeros(SUB)

for i in xrange(1, SUB + 1):
    rf_mnist_important_sub = rf_mnist_important.sub_randomForest(sub_size = i * MAX//SUBS)
    rf_mnist_important_sub = rf_mnist_important_sub.fit(mnist_X_train[:, mnist_important_features], mnist_y_train)
    important_train_error[i - 1] = rf_mnist_important_sub.compute_error(mnist_X_train[:, mnist_important_features], mnist_y_train)
    important_test_error[i -1] = rf_mnist_important_sub.compute_error(mnist_X_test[:, mnist_important_features], mnist_y_test)
    important_oob_error[ i -1] = rf_mnist_important_sub.oobe()

num_of_trees = np.arange(1, SUB + 1) * (MAX//SUBS)

plt.plot(num_of_trees, important_train_errors, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, important_oob_errors, 'b*-', linewidth = 1.5, label = 'out-of-bag error')
plt.plot(num_of_trees, important_test_errors, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('MNIST, random forest, 200 features, trees max depth = 5', fontsize = 16)
plt.show()


# 4.1.2. office
#     instead of resample from the training set, we adjust the weight for each
#         sample according to its label, that is, for samples with label $k$,
#         we multiply a factor of $(1/K) * (N/N_k)$, where $N_k$ is the number of
#         samples with label k.

MAX = 200
SUBS = 10
boost_office_balanced = ml.AdaBoostClassifier(n_estimators = MAX, max_depth = 10,
                                        class_weight = 'auto',
                                        random_state = 5)
boost_office_balanced = boost_office_balanced.fit(office_X_train, office_y_train)

ob_train_errors_b = np.ones(SUBS)
ob_test_errors_b = np.ones(SUBS)

for i in xrange(1, SUBS+1):
    sub_ada = boost_office_balanced.sub_adaboost(i * MAX//SUBS)
    ob_train_errors_b[i-1] = sub_ada.compute_error(office_X_train, office_y_train)
    ob_test_errors_b[i-1] = sub_ada.compute_error(office_X_test, office_y_test)

num_of_trees = np.arange(1, SUBS + 1) * (MAX//SUBS)

plt.plot(num_of_trees, ob_train_errors_b, 'rs-', linewidth = 1.5, label = 'train error')
plt.plot(num_of_trees, ob_test_errors_b, 'ko-', linewidth = 1.5, label = 'test error')
plt.xlabel('number of trees', fontsize = 16)
plt.ylabel('error rate', fontsize = 16)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('office, AdaBoost, training set balanced, max depth 10', fontsize = 16)
plt.show()

sub_ada = boost_office_balanced.sub_adaboost(MAX//SUBS)
sub_ada_cm_b= confusion_matrix(office_y_test, sub_ada.predict(office_X_test))
ada_office_b_cm = confusion_matrix(office_y_test, boost_office_balanced.predict(office_X_test))

ml.plot_cm(sub_ada_cm_b, labels = office_labels,
    title = 'confusion matrix, AdaBoost, training set balanced, max depth 10, 10 trees')
ml.plot_cm(ada_office_b_cm, labels = office_labels,
    title = 'confusion matrix, AdaBoost, training set balanced, max depth 10, 100 trees')
