#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/4/2023

Author: Alexander Roocroft
"""

import csv
import pandas as pd

import matplotlib.pyplot as plt
import pickle


df = pd.read_csv('Szep_etal_2022_MFQsurvey_data.csv')

print(df.head())
print(df.columns)

column_names = df.columns.tolist()
print(column_names)

mfq_df= df[['mfq_care', 'mfq_fairness', 'mfq_ingroup', 'mfq_authority', 'mfq_purity']].copy()

mfq_df['Care'] = mfq_df['mfq_care'].apply(lambda x: 1 if x > 23 else 0)
mfq_df['Fairness'] = mfq_df['mfq_fairness'].apply(lambda x: 1 if x > 23 else 0)
mfq_df['Ingroup'] = mfq_df['mfq_ingroup'].apply(lambda x: 1 if x > 23 else 0)
mfq_df['Authority'] = mfq_df['mfq_authority'].apply(lambda x: 1 if x > 23 else 0)
mfq_df['Purity'] = mfq_df['mfq_purity'].apply(lambda x: 1 if x > 23 else 0)

new_mfq_df = mfq_df.iloc[::16, :].copy()

new_mfq_df.drop(columns=['mfq_care', 'mfq_fairness', 'mfq_ingroup', 'mfq_authority', 'mfq_purity'], inplace=True)

combinations = new_mfq_df.groupby(['Care', 'Fairness', 'Ingroup', 'Authority', 'Purity']).size().reset_index(name='count')
total_rows = new_mfq_df.shape[0]
combinations['percentage'] = (combinations['count'] / total_rows) * 100
total_percentage = combinations['percentage'].sum()

# Using percentages as probabilities
combinations['moral_probabilities'] = combinations['percentage'] / 100
moral_probabilities= combinations['percentage'] / 100


rows_with_all_zeros = new_mfq_df[(new_mfq_df['Care'] == 0) & (new_mfq_df['Fairness'] == 0) & (new_mfq_df['Ingroup'] == 0) & (new_mfq_df['Authority'] == 0) & (new_mfq_df['Purity'] == 0)]
percentage_all_zeros = (rows_with_all_zeros.shape[0] / new_mfq_df.shape[0]) * 100

mfq_counts = new_mfq_df.eq(1).sum()
no_overall_mfq_count = len(new_mfq_df[new_mfq_df.eq(0).all(axis=1)])
mfq_counts['No Moral'] = no_overall_mfq_count

moral_entries={}
moral_entries[0] = len(new_mfq_df[new_mfq_df.sum(axis=1) == 0])
moral_entries[1] = len(new_mfq_df[new_mfq_df.sum(axis=1) == 1])
moral_entries[2]  = len(new_mfq_df[new_mfq_df.sum(axis=1) == 2])
moral_entries[3]  = len(new_mfq_df[new_mfq_df.sum(axis=1) == 3])
moral_entries[4]  = len(new_mfq_df[new_mfq_df.sum(axis=1) == 4])
moral_entries[5]  = len(new_mfq_df[new_mfq_df.sum(axis=1) == 5])
# total_moral_entries= sum(moral_entries.values())

# Extract labels and values from the dictionary
labels = list(moral_entries.keys())
values = list(moral_entries.values())
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('Number of Dummys per user')  # Set the title for each pie chart

plt.axis('equal')
plt.savefig('profile_1.pdf', format='pdf')
plt.show()

counts = pd.Series(mfq_counts)
labels = counts.index
percentage_profile=counts/len(new_mfq_df)


# Calculate the number of columns required for the arrangement
num_columns = 2
num_rows = len(percentage_profile) // num_columns + (len(percentage_profile) % num_columns > 0)

# Creating a figure with subplots
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 6))

# Flatten the subplots axes if necessary
if num_rows > 1:
    axs = axs.flatten()

# Plotting the pie charts
for i, (category, percentage) in enumerate(percentage_profile.items()):
    values = [percentage, 1 - percentage]  # Values for the pie chart
    labels = [category, 'Other']  # Labels for the pie chart

    axs[i].pie(values, labels=labels, autopct='%1.1f%%')
    axs[i].set_aspect('equal')  # Set aspect ratio to be equal

    axs[i].set_title(category)  # Set the title for each pie chart

# Hide unused subplots
for j in range(len(percentage_profile), len(axs)):
    axs[j].axis('off')

# Adjust the spacing between subplots
fig.tight_layout()

# Display the figure
plt.savefig('profile_2.pdf', format='pdf')
plt.show()


with open('moral_profile.pickle', 'wb') as file:
    pickle.dump(
        (combinations,moral_probabilities), file)

combinations.to_csv('moral_profile.csv', index=False)

