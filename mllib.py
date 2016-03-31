# Author: Xin Wang
# Email: xinwangmath@gmail.com

import numpy as np
import scipy as sp
import scipy.linalg
from sklearn import tree
#import pydot
#from IPython.display import Image
from sklearn.externals.six import StringIO
import operator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#def confusion_matrix(y_true, y_pred, labels = None):

#    if labels is None:
#        labels = np.unique(np.concatenate(y_true, y_pred))

    #p = len(labels)

def plot_cm(cm, labels, title = 'confusion matrix',  cmap = plt.cm.Blues):
    """plot the confusion matrix
    arguments:
       cm: confusion_matrix, a numpy 2d array
       labels: a list of labels, in sorted order
       title: default = 'confusion matrix'
       cmap: color map, default = plt.cm.Blues
    return: None
    side effect: plot the confusion_matrix
    """
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('true value')
    plt.xlabel('predicted value')

def split_dataset(X, y, training_ratio = 0.75, random_state = None):
    """split a dataset into training and testing datasets with training_ratio * n_samples
        in the training dataset
       arguments:
       X -- a N by p dataset, stored as a numpy array
       y -- a size N numpy vector storing the labels
       training_ratio -- a float number between 0 and 1, default = 0.75
       random_state -- random_seed, None or an int

       return:
       X_train, y_train, X_test, y_test
    """
    if training_ratio > 1 or training_ratio < 0:
        raise ValueError('mllib.split_dataset(): raining_ratio should be between 0 and 1')

    if len(X.shape) == 1:
        N = 1
    else:
        N = X.shape[0]

    assert(y.shape[0] == N)

    if random_state is None:
        pass
    else:
        np.random.seed(random_state)

    training_size = np.floor(N * training_ratio)
    tot_index = np.arange(N)
    np.random.shuffle(tot_index)
    training_index = tot_index[:training_size]
    testing_index = tot_index[training_size:]

    return X[training_index, :], y[training_index], X[testing_index, :], y[testing_index]

def split_dataset_index(tot_index, training_ratio = 0.75, random_state = None):
    """split a dataset into training and testing datasets with training_ratio * n_samples
        in the training dataset, return indices
       arguments:
       tot_index -- array of original dataset indices or an int
                 if it is an int, the tot_index array is set to np.arange(tot_index)
       training_ratio -- a float number between 0 and 1, default = 0.75
       return:
       training_index, testing_index
    """
    if training_ratio > 1 or training_ratio < 0:
        raise ValueError('mllib.split_dataset_index(): raining_ratio should be between 0 and 1')

    if random_state is None:
        pass
    else:
        np.random.seed(random_state)

    if isinstance(tot_index, int):
        tot_index = np.arange(tot_index)

    N = tot_index.shape[0]
    training_size = np.floor(N * training_ratio)

    np.random.shuffle(tot_index)

    return tot_index[:training_size], tot_index[training_size:]


def resample(X, y = None, n_samples = None, replace = True, random_state = None):
    """argument: X -- a N by p data matrix, stored as a numpy array
                 y -- a size N numpy array, optional
                 n_samples -- a int, float or None, optional, default = None
                              if None, n_samples = X.shape[0]
                 repalce -- boolean True or False, default = True
                 random_state -- optional
        return: if y is None, X_sample, and sample_index
                otherwise, X_sample, y_sample and sample_index
    """
    tot_size = X.shape[0]

    if n_samples is None:
        n_samples = tot_size
    elif isinstance(n_samples, int):
        if n_samples > tot_size:
            raise ValueError('mllib.resample(): n_samples can not be greater than the size of the dataset.')
        else:
            pass
    elif isinstance(n_samples, float):
        if n_samples > 1:
            raise ValueError('mllib.resample(): n_samples can not be greater than the size of the dataset.')
        else:
            n_samples = np.floor(n_samples * tot_size)
    else:
        raise TypeError('mllib.resample(): n_samples should be an int or a float number')

    if not isinstance(replace, bool):
        raise TypeError('mllib.resample(): replace should be boolean variable')

    if random_state is None:
        pass
    else:
        np.random.seed(random_state)

    if y is None:
        if replace:
            sample_index = np.random.randint(tot_size, size = n_samples)
            X_sample = X[sample_index, :]
            return X_sample, sample_index
        else:
            sample_index = np.arange(tot_size)
            np.random.shuffle(sample_index)
            X_sample = X[sample_index[:n_samples], :]
            return X_sample, sample_index[:n_samples]
    else:
        if y.shape[0] != tot_size:
            raise ValueError('mllib.resample(): size of X and y don\'t match')

        if replace:
            sample_index = np.random.randint(tot_size, size = n_samples)
            X_sample = X[sample_index, :]
            y_sample = y[sample_index]
            return X_sample, y_sample, sample_index
        else:
            sample_index = np.arange(tot_size)
            np.random.shuffle(sample_index)
            X_sample = X[sample_index[:n_samples], :]
            y_sample = y[sample_index[:n_samples]]
            return X_sample, y_sample, sample_index[:n_samples]

class RandomForestClassifier:
    """RandomForestClassifier(n_trees = 10, criterion = 'gini', max_depth = None,
        min_samples_each_leaf = 1,max_features = 'auto', random_state = None):
        the arugment list minics sklearn.ensemble.RandomForestClassifier class
        n_trees -- number of decision trees used
        criterion -- spliting criterion, 'gini' or 'maxinfogain', default = 'gini'
        max_depth -- the max depth for each decision tree, if None, then grow to
                     full extent, defalut = None
        min_samples_each_leaf -- the min number of samples in each leaf node,
                                 default = 1
        max_features -- number of features used n at each split,
                        'auto', 'sqrt', 'log2', 'None' or an int
                        'auto' = 'sqrt', max_features = sqrt(n_features)
                        'log2', max_features = log2(n_features)
                        None: max_features = n_features
                        int, max_features = entered integer
        # bootstrap -- True (use bootstrapped dataset) or False (use original dataset)
        # oob_score -- True or False
        random_state -- random seed """
    def __init__(self, n_trees = 10, criterion = 'gini', max_depth = None,
        min_samples_each_leaf = 1, max_features = 'auto', random_state = None):

        self.n_trees = n_trees;
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_each_leaf = min_samples_each_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.forest = [0] * n_trees
        self.oob_error = None
        self.oobe_list = [0] * n_trees
        self.feature_importances = None


    def fit(self, X, y, sample_weight = None):
        """fit a random forest to (X, y) with sample_weight
           return self"""
        N = X.shape[0]
        n_features = X.shape[1]
        assert(y.shape[0] == N)

        if self.random_state is None:
            pass
        else:
            np.random.seed(self.random_state)

        oob_pred_dict = [{} for _ in range(N)]

        for b in range(self.n_trees):
            random_state_b = None if self.random_state is None else np.random.randint(self.n_trees * 10)
            X_b, y_b, index_b = resample(X, y, random_state = random_state_b)
            sample_weight_b = None if sample_weight is None else sample_weight[index_b]

            T_b = tree.DecisionTreeClassifier(criterion = self.criterion, max_depth = self.max_depth,
                min_samples_leaf = self.min_samples_each_leaf, max_features = self.max_features,
                random_state = random_state_b)
            T_b = T_b.fit(X_b, y_b, sample_weight_b)

            self.forest[b] = T_b

            # out-of-bag error computation:
            #  record the predictions from this tree, increment the out-of-bag predction dict
            oob_sample_index = list(set(range(N)) - set(index_b))
            oob_pred = T_b.predict(X[oob_sample_index, :])
            for i in xrange(len(oob_sample_index)):
                if oob_pred[i] in oob_pred_dict[oob_sample_index[i]]:
                    oob_pred_dict[oob_sample_index[i]][oob_pred[i]] += 1
                else:
                    oob_pred_dict[oob_sample_index[i]][oob_pred[i]] = 1

            # compute the oob error with the current out-of-bag prediction dict
            self.oobe_list[b] = self._compute_oobe(N, oob_pred_dict, y)

        self.oob_error = self.oobe_list[-1]
        self.compute_feature_importances()

        return self

    def _compute_oobe(self, N, oob_pred_dict, y):
        pred_aggregate = [max(d.iteritems(), key = operator.itemgetter(1))[0]
            if len(d) > 0 else None for d in oob_pred_dict]
        pred_aggregate = np.array(pred_aggregate)

        oobe = float(N - np.sum(pred_aggregate == y))/N

        return oobe

    def oobe(self):
        return self.oob_error

    def get_oobe_list(self):
        return self.oobe_list

    def _set_oobe(self, oobe):
        self.oob_error = oobe

    def _set_oobe_list(self, new_list):
        for b in xrange(self.n_trees):
            self.oobe_list[b] = new_list[b]

    def _set_forest(self, new_forest):
        for b in xrange(self.n_trees):
            self.forest[b] = new_forest[b]

    def predict(self, X):
        """give prediction of labels for rows of X
        arguments:
           X -- a M by p new data matrix, stored as a numpy array
        return:
           an numpy array of predicted labels
        """
        if len(X.shape) == 1:
            M = 1
        else:
            M = X.shape[0]

        pred_dict = [{} for _ in range(M)]

        def pred_increment(pred_dict, new_pred):
            for i in xrange(len(pred_dict)):
                if new_pred[i] in pred_dict[i]:
                    pred_dict[i][new_pred[i]] += 1
                else:
                    pred_dict[i][new_pred[i]] = 1

        for b in range(self.n_trees):
            prediction_b = self.forest[b].predict(X)
            pred_increment(pred_dict, prediction_b)

        #print pred_dict
        pred_aggregate = [max(d.iteritems(), key = operator.itemgetter(1))[0] for d in pred_dict]
        return np.array(pred_aggregate)

    def compute_error(self, X, y):
        """arguments:
           X -- new N by p data matrix, stored as a numpy array
           y -- new label vector, stored as a size N numpy array
           return:
           error of the random forest prediction"""
        N = len(y)
        pred = self.predict(X)
        error = float(N - np.sum(pred == y))/N
        return error

    def sub_randomForest(self, sub_size):
        """returns a sub_randomForest of sub_size by picking the first sub_size trees
        from the current forest"""

        if sub_size > self.n_trees:
            raise ValueError('mllib.RandomForestClassifier.sub_randomForest(): sub_size can not be larger than current size')

        sub_rf = RandomForestClassifier(n_trees = sub_size,
                                        criterion = self.criterion,
                                        max_depth = self.max_depth,
                                        min_samples_each_leaf = self.min_samples_each_leaf,
                                        max_features = self.max_features,
                                        random_state = self.random_state)
        sub_rf._set_oobe(self.oobe_list[sub_size-1])
        sub_rf._set_oobe_list(self.oobe_list[:sub_size])
        sub_rf._set_forest(self.forest[:sub_size])
        sub_rf.compute_feature_importances()

        return sub_rf

    def compute_feature_importances(self):
        """compute the feature importances"""
        self.feature_importances = np.zeros(len(self.forest[0].feature_importances_))
        for i in xrange(self.n_trees):
            self.feature_importances = self.feature_importances + self.forest[i].feature_importances_

        self.feature_importances = self.feature_importances/self.n_trees

    def get_feature_importances(self):
        return self.feature_importances


class AdaBoostClassifier:
    """
    AdaBoostClassifier(n_estimators = 50, criterion = 'gini', max_depth = 2,
        max_features = 'auto', random_state = None):
    n_estimators -- number of weak estimators used, also the number of iterations;
    criterion -- spliting criterion for the decision tree, 'gini' or 'entropy';
    max_depth -- max depth for the decision tree, None or int, if None, then tree
                 grows to full extent
    max_features -- number of features used in each split, 'auto', 'sqrt', 'log2',
                        None or int
                    'auto' = 'sqrt', use sqrt(n_features) in each split
                    'log2', use log2(n_features) in each split
                    None, use n_features in each split
                    int, use the specified number of features in each split
    class_weight -- optional, default = None
                    if None, then class_weight is not used
                    if 'auto', then each sample is reweighted by a factor of
                    $(1/K) * (N/N_k)$, where $k$ is the true label of this sample,
                    $N_k$ is the number of samples with label k.
                    if a dict (label: weight), then the dict is taken as class_weight
    random_state -- random_seed
    """

    def __init__(self, n_estimators = 50, criterion = 'gini', max_depth = 1,
        max_features = 'auto', class_weight = None, random_state = None):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight

        self.weak_learners = [0] * n_estimators
        self.alpha = np.zeros(n_estimators)
        self.err = np.zeros(n_estimators)

    def _set_weak_learners(self, learners):
        for i in xrange(self.n_estimators):
            self.weak_learners[i] = learners[i]

    def _set_alpha(self, alpha):
        for i in xrange(self.n_estimators):
            self.alpha[i] = alpha[i]

    def _set_err(self, err):
        for i in xrange(self.n_estimators):
            self.err[i] = err[i]

    def sub_adaboost(self, sub_size):
        """return a sub adaboost aggregation of sub_size with the same parameters
        except the n_estimators = sub_size"""
        if sub_size > self.n_estimators:
            raise ValueError('AdaBoostClassifier.sub_adaboost(): sub_size cannot be greater than n_estimators')

        sub_ada = AdaBoostClassifier(n_estimators = sub_size,
                                     criterion = self.criterion,
                                     max_depth = self.max_depth,
                                     max_features = self.max_features,
                                     class_weight = self.class_weight,
                                     random_state = self.random_state)
        sub_ada._set_weak_learners(self.weak_learners[:sub_size])
        sub_ada._set_alpha(self.alpha[:sub_size])
        sub_ada._set_err(self.err[:sub_size])

        return sub_ada

    def fit(self, X, y):

        if len(X.shape) == 1:
            N = 1
            n_features = len(X)
        else:
            N = X.shape[0]
            n_features = X.shape[1]
        assert(y.shape[0] == N)

        if self.random_state is None:
            pass
        else:
            np.random.seed(self.random_state)

        # initialization
        weight = np.ones(N) * 1.0/N

        for m in xrange(self.n_estimators):
            random_state_m = None if self.random_state is None else np.random.randint(self.n_estimators * 10)
            T_m = tree.DecisionTreeClassifier(criterion = self.criterion,
                max_depth = self.max_depth, max_features = self.max_features,
                class_weight = self.class_weight, random_state = random_state_m)
            T_m = T_m.fit(X, y, sample_weight = weight)
            self.weak_learners[m] = T_m

            pred_m = T_m.predict(X)
            self.err[m] = np.sum(weight * (y != pred_m))/np.sum(weight)

            if self.err[m] > 0:
                self.alpha[m] = np.log( (1 - self.err[m])/self.err[m] ) + np.log(n_features - 1)
                weight = weight * np.exp( (y != pred_m) * self.alpha[m] )
                weight = weight/np.sum(weight)
            else:
                self.alpha[m] = np.log( (1 - 1e-12)/(1e-12) ) + np.log(n_features - 1)
                #print 'zero'

        return self

    def predict(self, X):
        """give prediction of labels for rows of X
        arguments:
          X -- a M by p new data matrix, stored as a numpy array
        return:
          an numpy array of predicted labels
        """
        if len(X.shape) == 1:
            M = 1
        else:
            M = X.shape[0]

        pred_dict = [{} for _ in range(M)]

        def pred_increment(pred_dict, new_pred, new_alpha):
            for i in xrange(len(pred_dict)):
                if new_pred[i] in pred_dict[i]:
                    pred_dict[i][new_pred[i]] += 1.0 * new_alpha
                else:
                    pred_dict[i][new_pred[i]] = 1.0 * new_alpha

        for m in xrange(self.n_estimators):
            prediction_m = self.weak_learners[m].predict(X)
            pred_increment(pred_dict, prediction_m, self.alpha[m])

        pred_aggregate = [max(d.iteritems(), key = operator.itemgetter(1))[0] for d in pred_dict]
        return np.array(pred_aggregate)

    def compute_error(self, X, y):
        """arguments:
           X -- new N by p data matrix, stored as a numpy array
           y -- new label vector, stored as a size N numpy array
           return:
           error of the adaboost prediction"""
        N = len(y)
        pred = self.predict(X)
        error = float(N - np.sum(pred == y))/N
        return error






















def main():
    #X = np.eye(10)
    #y = np.arange(10)

    #X_s, y_s, index_s = resample(X, y, random_state = None)

    #print X_s
    #print y_s
    #print index_s

    from sklearn.datasets import load_iris
    iris = load_iris()
    rf = RandomForestClassifier()
    rf = rf.fit(iris.data, iris.target)
    print rf.forest

    labels = rf.predict(iris.data)
    print labels
    print iris.target
    print rf.oobe()
    print rf.get_oobe_list()

    print rf.compute_error(iris.data, iris.target)

    sub_rf = rf.sub_randomForest(5)
    print sub_rf.oobe()
    print sub_rf.get_oobe_list()

    sub_pred = sub_rf.predict(iris.data)
    print sub_pred

    adaboost = AdaBoostClassifier(n_estimators = 10, max_depth = 2, random_state = 0)
    adaboost = adaboost.fit(iris.data, iris.target)

    labels = adaboost.predict(iris.data)
    print "adaboost"
    print labels
    print iris.target

    print adaboost.compute_error(iris.data, iris.target)

    print "test"
    N = 10
    train, test = split_dataset_index(N, random_state = 0)
    print train
    print test


if __name__ == '__main__':
    main()
