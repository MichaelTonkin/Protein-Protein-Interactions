#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
This script is used to train a SVM classifier. It takes the energy based metric data (complex_normalized and dG_cross) as input, and predicts if two proteins can interact.
'''
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from dataset import get_data
#from pre_process_data import structure_score_LHD, energy_score_LHD, join_data_LHD
from Pre_processing_data import some_data, less_data, more_data
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

def linear_regression_classification(energy_met, struct_met, join_data):
    # energy_met is the energy metrics
    # struct_met is the structure metrics
    # create a variable that contains both the energy and structure
    metrics = list(energy_met + struct_met)
    
    # split the data into train and test, x is the input, y is the output.
    x_train, x_test, y_train, y_test = train_test_split(join_data[metrics], join_data['label'], test_size=0.25,
                                                        random_state = 0, shuffle = True)
    
    # define the model
    clf = SVC(kernel='linear', random_state = 0)

    # fit the training data in the model to be trained
    clf.fit(x_train, y_train)

    # predict the label
    y_pred = clf.predict(x_test)
    
    # calculate the accuracy
    acc = accuracy_score(y_test, y_pred)

    # print the accuracy
    print(f'Accuracy score for {struct_met[:]} and {energy_met[:]} is', acc)
    return acc
    

# In[2]:


from itertools import combinations
import numpy as np

# list of labels
energy_labels = ['dG_separated', 'dG_separated/dSASAx100', 'dSASA_hphobic', 'dSASA_int', 'dSASA_polar', 'delta_unsatHbonds', 'hbond_E_fraction',
                  'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbonds_int', 'per_residue_energy_int', 'shape_complementarity']
struct_labels = ['ptm', 'iptm']


# create the different combinations of the parameters
for x in range(1, len(struct_labels) + 1):
    for struct_combo in combinations(struct_labels, x):
        for y in range(1, len(energy_labels) + 1):
            for energy_combo in combinations(energy_labels, y):
                
                # call the function that calculate the accuracy of the predicted data
                acc_val = linear_regression_classification(list(energy_combo), list(struct_combo),some_data)
                
                # linear_regression_classification(list(energy_combo), list(struct_combo),less_data) # use a different table that contains more data than some_data
                # linear_regression_classification(list(energy_combo), list(struct_combo),more_data) # there are 32 negative in more_data




# In[15]:


# Import KFold from sklearn.model_selection
from sklearn.model_selection import KFold

# Number of splits
num_sp = [2,3,4,5,6,7,8,9]

# Inititalise a matrix to store the test accuracies of different splits
accuracy = [[] for _ in range(len(num_sp))]

# use 3 energy metrics
metrics = ['ptm', 'dG_separated', 'dG_separated/dSASAx100']

# save the table of the data
join_data = some_data

# Loop over the values of k: 
for i,k in enumerate(num_sp):
    # Instantiate KFold with different size of splits. 
    # Set the parameter random_state to help reproduce the results.
    cv = KFold(n_splits = k, random_state = 0, shuffle = True)

    # Instantiate SVC classifier (Use the sklearn classifier) 
    clf = SVC(kernel='linear', random_state = 0)
    
    # Loop over the cross-validation splits: 
    for train_index, test_index in cv.split(join_data):
        train = join_data.loc[train_index]
        test = join_data.loc[test_index]

        x_train = train[metrics]
        y_train = train[['label']]
        x_test = test[metrics]
        y_test = test[['label']]
        
        # fit the model on the current split of data 
        clf.fit(x_train, y_train)
        
        # make predictions 
        y_pred = clf.predict(x_test)
        
        # calculate test accuracy and store 
        accuracy[i].append(accuracy_score(y_test,y_pred.T))

# Calculate the mean test accuracies across different number of splits
mean_arr = []
for i in range(len(num_sp)):
    mean_arr.append(np.mean(accuracy[i]))

# plot the different values of accuracy to see how it changes
plt.plot(num_sp, mean_arr,'.-')
plt.xlabel('Number of splits in the KFold')
plt.ylabel('Value of accuracy')


# In[ ]:




