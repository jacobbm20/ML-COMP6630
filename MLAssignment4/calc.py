import pandas as pd


train_file = "train.csv"
test_file = "test.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

def get_feature_types(df):
    discrete_features = []
    continuous_features = []
    
    for column in df.columns:
        if df[column].dtype == 'object':
            discrete_features.append(column)
        elif df[column].nunique() < 10: 
            discrete_features.append(column)
        else: 
            continuous_features.append(column)
    
    return discrete_features, continuous_features

# Get discrete and continuous features
discrete_features, continuous_features = get_feature_types(train)

# Conditional Probability Distributions

# Discrete Features
# Calculate the conditional probability distribution for discrete columns
def get_discrete_probability(discrete_column, target_column):
    # Group by the discrete column and target column
    distribution_counts = train.groupby([discrete_column, target_column]).size().unstack(fill_value=0)
    
    # Normalize the counts to get probabilities
    conditional_probabilities = distribution_counts.div(distribution_counts.sum(axis=1), axis=0)
    # View the table distribution counts
    print("Distribution Counts:")
    print(distribution_counts)

    # View the conditional probability distribution
    print("\nConditional Probability Distribution:")
    print(conditional_probabilities)
    


def get_continous_probability(continuous_column, column):

    def get_bin(column, num_bins=3):
      column_values = train[column]
      column_values_sorted = column_values.sort_values(ascending=True)
      bin_size = len(column_values_sorted) // num_bins
      bins = [column_values_sorted.iloc[i * bin_size:(i + 1) * bin_size].tolist() for i in range(num_bins)]
      
      # Handle any remaining values (if the length isn't perfectly divisible by num_bins)
      if len(column_values_sorted) % num_bins != 0:
          bins[-1].extend(column_values_sorted.iloc[num_bins * bin_size:].tolist())
      print(f"Bin 1: {bins[0]}")
      print(f"Bin 2: {bins[1]}")
      print(f"Bin 3: {bins[2]}")
      # get range of each bin
      bin_ranges = []
      for bin in bins:
          min_value = min(bin)
          max_value = max(bin)
          bin_ranges.append((min_value, max_value))
      return bin_ranges
    num_bins = 3
    bins = get_bin(column, num_bins=num_bins)
    # Create bins using pd.qcut for equal-sized bins
    bin_labels = ["Low", "Medium", "High"]
    train[f"{column} bin"] = pd.qcut(train[column], q=num_bins, labels=bin_labels)

    # Calculate the conditional probability distribution
    distribution_counts = train.groupby(["Construction type", f"{column} bin"], observed=True).size().unstack(fill_value=0)
    conditional_probabilities = distribution_counts.div(distribution_counts.sum(axis=1), axis=0)

    # View the table distribution counts
    print("Distribution Counts:")
    print(distribution_counts)

    # View the conditional probability distribution
    print("\nConditional Probability Distribution:")
    print(conditional_probabilities)

# Conditional Probabilities for discrete features
# Probabiliy of feature given 'Construction type'

# for column in discrete_features:
#   print(f"Conditional Probability Distribution for {column}:")
#   get_discrete_probability("Construction type", column)
#   print("___________________________________________________")
#   print("\n")

# for column in continuous_features:
#   print(f"Conditional Probability Distribution for {column}:")
#   get_continous_probability("Construction type", column)
#   print("___________________________________________________")
#   print("\n")   


# Function to calculate and return discrete probabilities as a dictionary
def get_discrete_probability_dict(discrete_column, target_column):
    # Group by the discrete column and target column
    distribution_counts = train.groupby([discrete_column, target_column]).size().unstack(fill_value=0)
    
    # Normalize the counts to get probabilities
    conditional_probabilities = distribution_counts.div(distribution_counts.sum(axis=1), axis=0)
    
    # Convert to dictionary format
    probabilities_dict = conditional_probabilities.to_dict(orient="index")
    
    # Print the dictionary
    print(f"P_{target_column} = {probabilities_dict}")
    return probabilities_dict

# Function to calculate and return continuous probabilities as a dictionary
def get_continous_probability_dict(continuous_column, column, num_bins=3):
    # Create bins using pd.qcut for equal-sized bins
    bin_labels = ["low", "mid", "high"]
    train[f"{column} bin"] = pd.qcut(train[column], q=num_bins, labels=bin_labels)
    
    # Group by the continuous column and the binned column
    distribution_counts = train.groupby([continuous_column, f"{column} bin"], observed=True).size().unstack(fill_value=0)
    
    # Normalize the counts to get probabilities
    conditional_probabilities = distribution_counts.div(distribution_counts.sum(axis=1), axis=0)
    
    # Convert to dictionary format
    probabilities_dict = conditional_probabilities.to_dict(orient="index")
    
    # Print the dictionary
    print(f"P_{column} = {probabilities_dict}")
    return probabilities_dict

# Example usage for discrete and continuous features
discrete_probabilities = {}
continuous_probabilities = {}

# Process discrete features
for column in discrete_features:
    print(f"Processing discrete feature: {column}")
    discrete_probabilities[column] = get_discrete_probability_dict("Construction type", column)

# Process continuous features
for column in continuous_features:
    print(f"Processing continuous feature: {column}")
    continuous_probabilities[column] = get_continous_probability_dict("Construction type", column)

# Print the final dictionaries
print("\nDiscrete Probabilities:")
print(discrete_probabilities)

print("\nContinuous Probabilities:")
print(continuous_probabilities)

# Apply MAP for Naive Bayes Classification


P_Bathrooms = {'Apartment': {1.0: 0.7142857142857143, 1.5: 0.14285714285714285, 2.5: 0.14285714285714285}, 'Condo': {1.0: 0.6666666666666666, 1.5: 0.16666666666666666, 2.5: 0.16666666666666666}, 'House': {1.0: 0.8571428571428571, 1.5: 0.14285714285714285, 2.5: 0.0}}

P_Garages = {'Apartment': {0.0: 0.14285714285714285, 1.0: 0.42857142857142855, 1.5: 0.14285714285714285, 2.0: 0.2857142857142857}, 'Condo': {0.0: 0.0, 1.0: 0.6666666666666666, 1.5: 0.0, 2.0: 0.3333333333333333}, 'House': {0.0: 0.2857142857142857, 1.0: 0.2857142857142857, 1.5: 0.14285714285714285, 2.0: 0.2857142857142857}}

P_Rooms = {'Apartment': {5: 0.14285714285714285, 6: 0.2857142857142857, 7: 0.2857142857142857, 8: 0.14285714285714285, 9: 0.14285714285714285, 10: 0.0}, 'Condo': {5: 0.0, 6: 0.6666666666666666, 7: 0.16666666666666666, 8: 0.0, 9: 0.0, 10: 0.16666666666666666}, 'House': {5: 0.14285714285714285, 6: 0.5714285714285714, 7: 0.2857142857142857, 8: 0.0, 9: 0.0, 10: 0.0}}

P_Bedrooms = {'Apartment': {2: 0.14285714285714285, 3: 0.42857142857142855, 4: 0.2857142857142857, 5: 0.14285714285714285}, 'Condo': {2: 0.0, 3: 0.8333333333333334, 4: 0.0, 5: 0.16666666666666666}, 'House': {2: 0.14285714285714285, 3: 0.7142857142857143, 4: 0.14285714285714285, 5: 0.0}}

P_Local_Price = {'Apartment': {'low': 0.5714285714285714, 'mid': 0.0, 'high': 0.42857142857142855}, 'Condo': {'low': 0.3333333333333333, 'mid': 0.3333333333333333, 'high': 0.3333333333333333}, 'House': {'low': 0.14285714285714285, 'mid': 0.5714285714285714, 'high': 0.2857142857142857}}

P_Land_Area = {'Apartment': {'low': 0.42857142857142855, 'mid': 0.2857142857142857, 'high': 0.2857142857142857}, 'Condo': {'low': 0.3333333333333333, 'mid': 0.3333333333333333, 'high': 0.3333333333333333}, 'House': {'low': 0.2857142857142857, 'mid': 0.2857142857142857, 'high': 0.42857142857142855}}

P_Living_area = {'Apartment': {'low': 0.42857142857142855, 'mid': 0.2857142857142857, 'high': 0.2857142857142857}, 'Condo': {'low': 0.5, 'mid': 0.3333333333333333, 'high': 0.16666666666666666}, 'House': {'low': 0.14285714285714285, 'mid': 0.42857142857142855, 'high': 0.42857142857142855}}

P_Age_of_home = {'Apartment': {'low': 0.2857142857142857, 'mid': 0.2857142857142857, 'high': 0.42857142857142855}, 'Condo': {'low': 0.3333333333333333, 'mid': 0.3333333333333333, 'high': 0.3333333333333333}, 'House': {'low': 0.8571428571428571, 'mid': 0.0, 'high': 0.14285714285714285}}

priors = {"Apartment": 7 / 20, "House": 7 / 20, "Condo": 6 / 20}

results = []

def bin_price(price):
    if price <= 5.06:
        return 'low'
    elif price <= 6:
        return 'mid'
    else:
        return 'high'
    
def bin_land_area(area):
    if area <= 4.5:
        return 'low'
    elif area <= 6.5:
        return 'mid'
    else:
        return 'high'

def bin_living_area(area):
    if area <= 1.122:
        return 'low'
    elif area <= 1.491:
        return 'mid'
    else:
        return 'high'

def bin_age_of_home(age):
    if age <= 30:
        return 'low'
    elif age <= 41:
        return 'mid'
    else:
        return 'high'

for _, row in test.iterrows():
    scores = {}
    price_bin = bin_price(row["Local Price"])
    land_area_bin = bin_land_area(row["Land Area"])
    living_area_bin = bin_living_area(row["Living area"])
    age_of_home_bin = bin_age_of_home(row["Age of home"])

    for cls in priors:
        try:
            # Calculate the score for each class
            scores[cls] = (
                priors[cls] *
                P_Bathrooms[cls].get(row["Bathrooms"], 1e-2) *
                P_Garages[cls].get(row["# Garages"], 1e-2) *
                P_Rooms[cls].get(row["# Rooms"], 1e-2) *
                P_Bedrooms[cls].get(row["# Bedrooms"], 1e-2) *
                P_Local_Price[cls].get(price_bin, 1e-2) *
                P_Land_Area[cls].get(land_area_bin, 1e-2) *
                P_Living_area[cls].get(living_area_bin, 1e-2) *
                P_Age_of_home[cls].get(age_of_home_bin, 1e-2)
            )
        except KeyError as e:
            print(f"KeyError: {e} for class {cls} and row {row}")
            scores[cls] = 0
    predicted = max(scores, key=scores.get)
    results.append(
        {
            "Local Price": f"{price_bin}: {row['Local Price']}",
            "Bathrooms": row["Bathrooms"],
            "Land Area": f"{land_area_bin}: {row['Land Area']}",
            "Living area": f"{living_area_bin}: {row['Living area']}",
            "# Garages": row['# Garages'],
            "# Rooms": row["# Rooms"],
            "# Bedrooms": row["# Bedrooms"],
            "Age of home": f"{age_of_home_bin}: {row['Age of home']}",
            "P(House)": (scores["House"]),
            "P(Apartment)": scores["Apartment"],
            "P(Condo)": scores["Condo"],
            "Actual Construction Type": row["Construction type"],
            "Predicted Class": predicted
            
        }
    )
# Create a DataFrame from the results
results_df = pd.DataFrame(results)
print(results_df)
# # Save the results to a CSV file
# results_df.to_csv("results.csv", index=False)
# print("Results saved to results.csv")