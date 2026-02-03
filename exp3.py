import pandas as pd
import numpy as np

# 1. Create the dataset (PlayTennis is the standard ID3 example)
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# 2. Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# 3. Function to calculate Information Gain
def InfoGain(data, split_attribute_name, target_name="Play"):
    total_entropy = entropy(data[target_name])
    vals, counts= np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    return total_entropy - Weighted_Entropy

# 4. Simplified ID3 function to show logic
def id3(data, original_data, features, target_attribute_name="Play"):
    # If all target values are the same, return that value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # Select the best feature
    item_values = [InfoGain(data, feature) for feature in features]
    best_feature_index = np.argmax(item_values)
    best_feature = features[best_feature_index]
    
    tree = {best_feature: {}}
    features = [i for i in features if i != best_feature]
    
    for value in np.unique(data[best_feature]):
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = id3(sub_data, df, features)
        tree[best_feature][value] = subtree
        
    return tree

# --- Run the Algorithm ---
features = list(df.columns[:-1])
tree = id3(df, df, features)

print("Generated Decision Tree:")
print(tree)

# 5. Classify a New Sample
# Sample: Outlook=Overcast, Temp=Hot, Humidity=High, Wind=Weak
def classify(sample, tree):
    for node in tree.keys():
        value = sample[node]
        tree = tree[node][value]
        if isinstance(tree, dict):
            return classify(sample, tree)
        else:
            return tree

new_sample = {'Outlook': 'Overcast', 'Temp': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
prediction = classify(new_sample, tree)
print(f"\nPrediction for New Sample {new_sample}: \nPlay = {prediction}")