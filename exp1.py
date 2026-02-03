import csv

def find_s_algorithm(data):
    # Initialize the specific hypothesis with the first positive example
    # We assume the last column is the target (Yes/No)
    attributes = data[0][:-1]
    hypothesis = ['0'] * len(attributes)

    print(f"Initial Hypothesis: {hypothesis}")

    for row in data:
        if row[-1].lower() == 'yes': # Only consider positive examples
            for i in range(len(attributes)):
                if hypothesis[i] == '0':
                    hypothesis[i] = row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i] = '?'
            print(f"Hypothesis after positive example: {hypothesis}")
    
    return hypothesis

# Sample Data: [Sky, Temp, Humidity, Wind, Water, Forecast, EnjoySport]
dataset = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

final_h = find_s_algorithm(dataset)
print(f"\nFinal Most Specific Hypothesis: {final_h}")