#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data from Data_IA.csv into a variable data
data = pd.read_csv("Data_IA.csv")

# Change the success value, from Yes to 1 and from No to 0
data['success'] = data['success'].replace(['Yes'],1)
data['success'] = data['success'].replace(['No'],0)

# rename the column 'success' to 'label'
data = data.rename(columns = {'success' : 'label'})

# Create a table that only includes the name of the model, its name of the alpha fold model, its category,
# its label and its structure scores.
structure_score = data[['NAME_NUMBER','AF_model','category','ptm','iptm','plddt','label']]

# Create a table that only includes the name of the model, its name of the alpha fold model, its category,
# its label and its energy scores.
energy_score = data[['NAME_NUMBER','AF_model','category','total_score','complex_normalized',
                     'dG_cross','dG_cross/dSASAx100','dG_separated','dG_separated/dSASAx100',
                     'dSASA_hphobic','dSASA_int','dSASA_polar','delta_unsatHbonds','dslf_fa13',
                     'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_intra_sol_xover4','fa_rep',
                     'fa_sol','hbond_E_fraction','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb',
                     'hbonds_int','lk_ball_wtd','nres_all','nres_int','omega','p_aa_pp','packstat',
                     'per_residue_energy_int','pro_close','rama_prepro','ref','shape_complementarity',
                     'side1_normalized','side1_score','side2_normalized','side2_score','yhh_planarity',
                     'H','E','L','hbond_scoverdSASA_int','hbond_scoverdsasa_int_magplusiptm','label']]

# Table containing only DHR design of structure scores.
structure_score_DHR = structure_score.query('category == "DHR"')
structure_score_DHR = structure_score_DHR[['NAME_NUMBER','ptm','iptm','plddt','label']]

# Table containing only LHD design of structure scores.
structure_score_LHD = structure_score.query('category != "DHR"')
structure_score_LHD = structure_score_LHD[['NAME_NUMBER','ptm','iptm','plddt','label']]

# Table containing only DHR design of energy scores.
energy_score_DHR = energy_score.query('category == "DHR"')
energy_score_DHR = energy_score_DHR[['NAME_NUMBER','total_score','complex_normalized',
                     'dG_cross','dG_cross/dSASAx100','dG_separated','dG_separated/dSASAx100',
                     'dSASA_hphobic','dSASA_int','dSASA_polar','delta_unsatHbonds','dslf_fa13',
                     'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_intra_sol_xover4','fa_rep',
                     'fa_sol','hbond_E_fraction','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb',
                     'hbonds_int','lk_ball_wtd','nres_all','nres_int','omega','p_aa_pp','packstat',
                     'per_residue_energy_int','pro_close','rama_prepro','ref','shape_complementarity',
                     'side1_normalized','side1_score','side2_normalized','side2_score','yhh_planarity',
                     'H','E','L','hbond_scoverdSASA_int','hbond_scoverdsasa_int_magplusiptm','label']]

# Table containing only LHD design of energy scores.
energy_score_LHD = energy_score.query('category != "DHR"')
energy_score_LHD = energy_score_LHD[['NAME_NUMBER','total_score','complex_normalized',
                     'dG_cross','dG_cross/dSASAx100','dG_separated','dG_separated/dSASAx100',
                     'dSASA_hphobic','dSASA_int','dSASA_polar','delta_unsatHbonds','dslf_fa13',
                     'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_intra_sol_xover4','fa_rep',
                     'fa_sol','hbond_E_fraction','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb',
                     'hbonds_int','lk_ball_wtd','nres_all','nres_int','omega','p_aa_pp','packstat',
                     'per_residue_energy_int','pro_close','rama_prepro','ref','shape_complementarity',
                     'side1_normalized','side1_score','side2_normalized','side2_score','yhh_planarity',
                     'H','E','L','hbond_scoverdSASA_int','hbond_scoverdsasa_int_magplusiptm','label']]

# table that contain all the data together with these columns, combine the data from the different categories
join_data = data[['NAME_NUMBER','AF_model','category','ptm','iptm','plddt','total_score','complex_normalized',
                    'dG_cross','dG_cross/dSASAx100','dG_separated','dG_separated/dSASAx100',
                    'dSASA_hphobic','dSASA_int','dSASA_polar','delta_unsatHbonds','dslf_fa13',
                    'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_intra_sol_xover4','fa_rep',
                    'fa_sol','hbond_E_fraction','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb',
                    'hbonds_int','lk_ball_wtd','nres_all','nres_int','omega','p_aa_pp','packstat',
                    'per_residue_energy_int','pro_close','rama_prepro','ref','shape_complementarity',
                    'side1_normalized','side1_score','side2_normalized','side2_score','yhh_planarity',
                    'H','E','L','hbond_scoverdSASA_int','hbond_scoverdsasa_int_magplusiptm','label']]

# table containing LHD category data                    
join_data_LHD = join_data.query('category == "LHD"')
join_data_LHD = join_data_LHD[['NAME_NUMBER','ptm','iptm','plddt','total_score','complex_normalized',
                    'dG_cross','dG_cross/dSASAx100','dG_separated','dG_separated/dSASAx100',
                    'dSASA_hphobic','dSASA_int','dSASA_polar','delta_unsatHbonds','dslf_fa13',
                    'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_intra_sol_xover4','fa_rep',
                    'fa_sol','hbond_E_fraction','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb',
                    'hbonds_int','lk_ball_wtd','nres_all','nres_int','omega','p_aa_pp','packstat',
                    'per_residue_energy_int','pro_close','rama_prepro','ref','shape_complementarity',
                    'side1_normalized','side1_score','side2_normalized','side2_score','yhh_planarity',
                    'H','E','L','hbond_scoverdSASA_int','hbond_scoverdsasa_int_magplusiptm','label']]

# table with all the data without 'category', 'AF_model' column
all_data = data[['NAME_NUMBER','ptm','iptm','plddt','total_score','complex_normalized',
                    'dG_cross','dG_cross/dSASAx100','dG_separated','dG_separated/dSASAx100',
                    'dSASA_hphobic','dSASA_int','dSASA_polar','delta_unsatHbonds','dslf_fa13',
                    'fa_atr','fa_dun','fa_elec','fa_intra_rep','fa_intra_sol_xover4','fa_rep',
                    'fa_sol','hbond_E_fraction','hbond_bb_sc','hbond_lr_bb','hbond_sc','hbond_sr_bb',
                    'hbonds_int','lk_ball_wtd','nres_all','nres_int','omega','p_aa_pp','packstat',
                    'per_residue_energy_int','pro_close','rama_prepro','ref','shape_complementarity',
                    'side1_normalized','side1_score','side2_normalized','side2_score','yhh_planarity',
                    'H','E','L','hbond_scoverdSASA_int','hbond_scoverdsasa_int_magplusiptm','label']]

# find all the positive data
possitive = all_data.query('label == 1')

# create a new table with only tha positive data
possitive_protein = possitive[['NAME_NUMBER']].to_numpy()[:,0].T

# find all the negative data
negative = all_data.query('label == 0')

# remove the negative data that have also positive value
for x in possitive_protein:
    negative = negative.query('NAME_NUMBER != @x')

# select 16 negatives
negative = negative.loc[0::11]

# create a new table that contains the positive at the top and negatives at the bottom
frames = [possitive, negative]
some_data = pd.concat(frames).reset_index(drop = True)
#some_data = some_data.sample(frac = 1,random_state = 0).reset_index(drop = True)

# create another table with more data than some_data
possitive_two = all_data.query('label == 1')

negative_two = all_data.query('label == 0')

for x in possitive_protein:
    negative_two = negative_two.query('NAME_NUMBER != @x')
negative_two = negative_two.loc[10::5]
frames_two = [possitive_two, negative_two]
more_data = pd.concat(frames_two).reset_index(drop = True)
#more_data = more_data.sample(frac = 1,random_state = 0).reset_index(drop = True)

# create another table with more data than some_data and less data than more_data
possitive_three = all_data.query('label == 1')

negative_three = all_data.query('label == 0')

for x in possitive_protein:
    negative_three = negative_three.query('NAME_NUMBER != @x')
negative_three = negative_three.loc[0::8]
frames_three = [possitive_three, negative_three]
less_data = pd.concat(frames_three).reset_index(drop = True)
#less_data = less_data.sample(frac = 1,random_state = 0).reset_index(drop = True)

