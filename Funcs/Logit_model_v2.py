# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:43:27 2023

@author: alexroocroft
"""

import numpy as np

import random


# ============================================================================
# %%  For Sacrifice-based scheme
def sacrifice_parameter_generation(number_of_users):
    """
    Generates user-specific parameters for a sacrifice-based social routing  scheme.

    Parameters:
        number_of_users (int): The number of users for whom parameters will be generated.

    Returns:
        dict: A dictionary where each key is a user ID, and the value is a dictionary of their parameters.
    """

    # Define parameter values for the Sacrifice-based scheme
    param_list_S = [
        5.49,  # ASC_SR (Alternative Specific Constant for Sacrifice-based Routing)
        [-0.811, 0.861],  # No_days: Mean and standard deviation for the number of days
        [-0.949, 0.737],  # Add_time: Mean and standard deviation for additional travel time
        [0.001, 0.06],  # Network_benefit: Mean and standard deviation for network benefit perception
        [0.017, 0.029],  # Participation_rate: Mean and standard deviation for participation likelihood
        2.047,  # Care coefficient
        0.807,  # Fairness coefficient
        -1.418,  # Ingroup coefficient
        0.819,  # Authority coefficient
        -1.083  # Purity coefficient
    ]

    # Define corresponding parameter names
    param_names = ['ASC_SR', 'No_days', 'Add_time', 'Network_benefit', 'Participation_rate',
                   'Care', 'Fairness', 'Ingroup', 'Authority', 'Purity']

    # Store parameters in a dictionary
    param_dict_store = dict(zip(param_names, param_list_S))

    user_param_S = dict()  # Dictionary to store individual user parameters

    for user_id in range(number_of_users):
        param_dict_user = dict()

        # Assign constant parameters
        param_dict_user['ASC_SR'] = param_dict_store['ASC_SR']
        param_dict_user['Care'] = param_dict_store['Care']
        param_dict_user['Fairness'] = param_dict_store['Fairness']
        param_dict_user['Ingroup'] = param_dict_store['Ingroup']
        param_dict_user['Authority'] = param_dict_store['Authority']
        param_dict_user['Purity'] = param_dict_store['Purity']

        # Assign normally distributed parameters based on mean and standard deviation
        param_dict_user["No_days"] = np.random.normal(param_dict_store["No_days"][0], param_dict_store["No_days"][1])
        param_dict_user["Add_time"] = np.random.normal(param_dict_store["Add_time"][0], param_dict_store["Add_time"][1])
        param_dict_user["Network_benefit"] = np.random.normal(param_dict_store["Network_benefit"][0],
                                                              param_dict_store["Network_benefit"][1])
        param_dict_user["Participation_rate"] = np.random.normal(param_dict_store["Participation_rate"][0],
                                                                 param_dict_store["Participation_rate"][1])

        # Store user-specific parameters
        user_param_S[user_id] = param_dict_user

    return user_param_S


# ============================================================================
# %%  For Collective Good-based scheme
def CollectiveGood_parameter_generation(number_of_users):
    """
    Generates user-specific parameters for a Collective Good-based social routing scheme.

    Parameters:
        number_of_users (int): The number of users for whom parameters will be generated.

    Returns:
        dict: A dictionary where each key is a user ID, and the value is a dictionary of their parameters.
    """

    # Define parameter values for the Collective Good-based scheme
    param_list_CG = [
        4.485,  # ASC_SR (Alternative Specific Constant for Collective Good-based Routing)
        [-0.698, 0.772],  # No_days: Mean and standard deviation for the number of days
        [-0.755, 0.932],  # Add_time: Mean and standard deviation for additional travel time
        [0.005, 0.051],  # Network_benefit: Mean and standard deviation for network benefit perception
        [0.021, 0.036],  # Participation_rate: Mean and standard deviation for participation likelihood
        0.479,  # Care coefficient
        1.34,  # Fairness coefficient
        -2.845,  # Ingroup coefficient
        2.184,  # Authority coefficient
        1.486  # Purity coefficient
    ]

    # Define corresponding parameter names
    param_names = ['ASC_SR', 'No_days', 'Add_time', 'Network_benefit', 'Participation_rate',
                   'Care', 'Fairness', 'Ingroup', 'Authority', 'Purity']

    # Store parameters in a dictionary
    param_dict_store = dict(zip(param_names, param_list_CG))

    user_param_CG = dict()  # Dictionary to store individual user parameters

    for user_id in range(number_of_users):
        param_dict_user = dict()

        # Assign constant parameters
        param_dict_user['ASC_SR'] = param_dict_store['ASC_SR']
        param_dict_user['Care'] = param_dict_store['Care']
        param_dict_user['Fairness'] = param_dict_store['Fairness']
        param_dict_user['Ingroup'] = param_dict_store['Ingroup']
        param_dict_user['Authority'] = param_dict_store['Authority']
        param_dict_user['Purity'] = param_dict_store['Purity']

        # Assign normally distributed parameters based on mean and standard deviation
        param_dict_user["No_days"] = np.random.normal(param_dict_store["No_days"][0], param_dict_store["No_days"][1])
        param_dict_user["Add_time"] = np.random.normal(param_dict_store["Add_time"][0], param_dict_store["Add_time"][1])
        param_dict_user["Network_benefit"] = np.random.normal(param_dict_store["Network_benefit"][0],
                                                              param_dict_store["Network_benefit"][1])
        param_dict_user["Participation_rate"] = np.random.normal(param_dict_store["Participation_rate"][0],
                                                                 param_dict_store["Participation_rate"][1])

        # Store user-specific parameters
        user_param_CG[user_id] = param_dict_user

    return user_param_CG


# ============================================================================
# %% Utility Calculation for social routing scheme
def utility_SR(index, row, user_param):
    """
    Computes the utility value for the social routing scheme.

    Parameters:
        index (int): The user index.
        row (dict-like): A row of data containing decision variables for a specific scenario.
        user_param (dict): Dictionary containing user-specific parameters.

    Returns:
        float: The computed utility value for the given user and scenario.
    """

    utility_SR1 = (
            user_param[index]["ASC_SR"] +
            row["SRdays_additional_time"] * user_param[index]["Add_time"] +
            row["Network_benefit"] * user_param[index]["Network_benefit"] +
            row["participation_rate"] * user_param[index]["Participation_rate"] +
            row["number_of_days"] * user_param[index]["No_days"] +
            row["Care"] * user_param[index]["Care"] +
            row["Fairness"] * user_param[index]["Fairness"] +
            row["Ingroup"] * user_param[index]["Ingroup"] +
            row["Authority"] * user_param[index]["Authority"] +
            row["Purity"] * user_param[index]["Purity"]
    )

    return utility_SR1


# ============================================================================
# %% Probability Calculation for social routing scheme
def prob_SR(utility_SR):
    """
    Computes the probability of selecting the social routing scheme.

    Parameters:
        utility_SR (float): The computed utility value for a user.

    Returns:
        float: The probability of selecting the social scheme.
    """

    e_user = np.exp(-1 * utility_SR)  # Compute the exponentiated negative utility
    prob_SR = 1 / (1 + e_user)  # Logistic transformation to get probability

    return prob_SR