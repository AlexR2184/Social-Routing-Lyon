# -*- coding: utf-8 -*-
"""
Created on 30/8/23

@author: alexroocroft
"""

# For file input/output
import pandas as pd
import pickle
import sys
import os

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to the Python path to provide access to Funcs folder
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Funcs import Logit_model_v2 as user_choices


# Cross-platform path handling
current_dir = os.getcwd()


# -------For the 4 hr Lyon simulation ----------

# --- Load Moral survey data ----
moral_input = os.path.join(current_dir, 'moral_profile.pickle')


with open(moral_input, 'rb') as file:
    combinations, moral_probabilities = pickle.load(file)

#-----Load SymuVia_Lyon_Inputs to obtain the number of users---------
SymuVia_Lyon_Inputs_path = os.path.join(current_dir, os.pardir, '1_SymuVia_Lyon_Inputs')
SymuVia_Lyon_Inputs_path = os.path.abspath(SymuVia_Lyon_Inputs_path)

demand_4h_path = os.path.join(SymuVia_Lyon_Inputs_path, '4_hour_demand_profile')

demand_input_init = os.path.join(demand_4h_path, 'UC_routes.csv')

veh_init = pd.read_csv(demand_input_init, sep=";")  # columns: origin;typeofvehicle;creation;path;destination

number_of_users = len(veh_init['id'])

# --- Moral profile generation for the users ----
user_combo = {}
for user_id in range(number_of_users):
    user_combo[user_id] = moral_probabilities.sample(n=1, weights=moral_probabilities).index[0]
moral_user = {}
for user_id in range(number_of_users):
    moral_user[user_id] = combinations[['Care', 'Fairness', 'Ingroup', 'Authority', 'Purity']].loc[
        user_combo[user_id]].to_dict()

# --- Logit coefficient profile ----

user_param_S = user_choices.sacrifice_parameter_generation(number_of_users)
user_param_CG = user_choices.CollectiveGood_parameter_generation(number_of_users)
user_param_store ={'Sacrifice' :user_param_S ,'CollectiveGood' :user_param_CG}


with open('user_profile.pickle', 'wb') as file:
        pickle.dump((user_param_store,moral_user), file)



# -------For the 2 hr Lyon simulation (1 hr loading) ----------

# ----- Load data from SymuVia_Lyon_Inputs to obtain the number of user ------
# Add another folder within SymuVia_Lyon_Inputs_path
demand_1h_path = os.path.join(SymuVia_Lyon_Inputs_path, '1_hour_demand_profile')

demand_input_init = os.path.join(demand_1h_path, 'init_routes_1h.csv')

veh_init = pd.read_csv(demand_input_init, sep=";")  # columns: origin;typeofvehicle;creation;path;destination

number_of_users = len(veh_init['id'])

#-----Create user_profile ---------
user_combo = {}
for user_id in range(number_of_users):
    user_combo[user_id] = moral_probabilities.sample(n=1, weights=moral_probabilities).index[0]
moral_user = {}
for user_id in range(number_of_users):
    moral_user[user_id] = combinations[['Care', 'Fairness', 'Ingroup', 'Authority', 'Purity']].loc[
        user_combo[user_id]].to_dict()

# --- Logit coefficient profile ----

user_param_S = user_choices.sacrifice_parameter_generation(number_of_users)
user_param_CG = user_choices.CollectiveGood_parameter_generation(number_of_users)
user_param_store ={'Sacrifice' :user_param_S ,'CollectiveGood' :user_param_CG}

#save as the 1 hour user parameter store.
with open('user_profile_1h.pickle', 'wb') as file:
        pickle.dump((user_param_store,moral_user), file)