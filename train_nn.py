#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Pre_processing_data import some_data, less_data, more_data
import matplotlib.pyplot as plt

def multilayer_neural_network_classification(energy_met, struct_met, join_data):
    # energy_met is the energy metrics
    # struct_met is the structure metrics
    # Combine the two metrics in one list
    metrics = list(energy_met + struct_met)

    # split the data into train and test
    # x is the input, y is the output.
    x_train, x_test, y_train, y_test = train_test_split(join_data[metrics], join_data['label'], test_size=0.25,
                                                        random_state = 0, shuffle = True)
    # define the model
    nn_mult = MLPClassifier(hidden_layer_sizes = (43,20), activation = 'relu', alpha = 1, max_iter = 10000, random_state = 0)

    # Train the model
    nn_mult.fit(x_train, y_train)

    # predict the label
    y_pred = nn_mult.predict(x_test)
    
    # calculate the accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy score for {struct_met[:]} and {energy_met[:]} is', acc)
    return acc


# In[2]:


from itertools import combinations
import numpy as np

#list of labels
energy_labels = ['dG_separated', 'dG_separated/dSASAx100', 'dSASA_hphobic', 'dSASA_int', 'dSASA_polar', 'delta_unsatHbonds', 'hbond_E_fraction',
                  'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbonds_int', 'per_residue_energy_int', 'shape_complementarity']
struct_labels = ['ptm', 'iptm']

# create the different combinations of the parameters
for x in range(1, len(struct_labels) + 1):
    for struct_combo in combinations(struct_labels, x):
        for y in range(1, len(energy_labels) + 1):
            for energy_combo in combinations(energy_labels, y):

                # call the function that calculate the accuracy of the predicted data
                multilayer_neural_network_classification(list(energy_combo), list(struct_combo),some_data)
                # multilayer_neural_network_classification(list(energy_combo), list(struct_combo),less_data) # use a different table that contains more data than some_data
                # multilayer_neural_network_classification(list(energy_combo), list(struct_combo),more_data) # there are 32 negative in more_data
                
