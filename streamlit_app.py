import streamlit as st
import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

# data_path = "/workspaces/growth_curve/data/"

# names = os.listdir(data_path)

# keys = [
#     'head',
#     'height',
#     'weight',
#     'weight_height'
# ]

# files = dict()

# for n in range(0,4):
#     files[keys[n]] = names[n]

def calculate_percentile(age, weight, df):
    """
    Calculate the percentile for a given age and weight using the LMS method.
    
    Parameters:
        age (float): Age in months.
        weight (float): Weight in kilograms.
        data (DataFrame): The dataset containing L, M, S values.
        
    Returns:
        float: The percentile rank.
    """
    # Find the closest age in the dataset
    row = df.loc[df['Month'] == age]
    if row.empty:
        return "Age not found in dataset. Please provide a valid age."

    # Extract L, M, S values
    L = row['L'].values[0]
    M = row['M'].values[0]
    S = row['S'].values[0]

    # Calculate the z-score
    if L != 0:
        z = ((weight / M) ** L - 1) / (L * S)
    else:
        z = np.log(weight / M) / S

    # Convert z-score to percentile
    percentile = norm.cdf(z) * 100

    return percentile

def calculate_weight_for_percentile(age, percentile, df):
    """
    Calculate the weight corresponding to a specific percentile for a given age.
    
    Parameters:
        age (float): Age in months.
        percentile (float): Desired percentile (e.g., 50 for 50th percentile).
        data (DataFrame): The dataset containing L, M, S values.
        
    Returns:
        float: The weight corresponding to the percentile.
    """
    percentile = float(percentile)
    # Find the closest age in the dataset
    row = df.loc[df['Month'] == age]
    if row.empty:
        return None

    # Extract L, M, S values
    L = row['L'].values[0]
    M = row['M'].values[0]
    S = row['S'].values[0]

    # Convert percentile to z-score
    z = norm.ppf(percentile / 100)

    # Calculate weight
    if L != 0:
        weight = M * ((1 + L * S * z) ** (1 / L))
    else:
        weight = M * np.exp(S * z)

    return weight

df = pd.read_csv('data/'+'weight.csv')

# Example usage
# age, weight = 13, 7.8
# age = input('how old is your child (in months)?: ')  # Age in months
# weight = input('how much does the child weight (in kilograms)?: ')  # Weight in kilograms

age = st.number_input("how old is your child (in months)?: ", min_value=0, max_value=60, value=0, step=1)
weight = st.number_input("how much does the child weight (in kilograms)?: ", min_value=0.0, max_value=100.0, value=0.0, step=0.1)


percentile = calculate_percentile(age, weight, df)
percentile_rounded = f"{percentile:.2f}"
res_msg = f"For a {age}-month-old girl weighing {weight} kg, the weight corresponds to the {percentile_rounded}th percentile."

st.markdown(f"<h3 style='font-size:30px;'>{percentile_rounded}th percentile</h3>", unsafe_allow_html=True)

st.write(res_msg)

percentile_to_plot = percentile_rounded

ages = df['Month']

weights = [round(calculate_weight_for_percentile(age, percentile_to_plot, df), 2) for age in ages]
weights = [w for w in weights if w is not None]

# Plot the curve
plt.figure(figsize=(15, 15))
plt.plot(ages, weights, label=f'{percentile_to_plot}th Percentile', color='blue')
plt.title("Weight-for-Age Percentile Curve", fontsize=14)
plt.xlabel("Age (Months)", fontsize=12)
plt.ylabel("Weight (kg)", fontsize=12)
plt.xticks(np.arange(ages.min(), ages.max() + 1, 1))  # Optional: Tick every 6 months
plt.yticks(np.arange(min(weights), max(weights) + 0.1, 0.2))  # Tick every 0.5 kg
plt.legend(fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the plot
st.pyplot(plt)