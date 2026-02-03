import csv

def candidate_elimination(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
        header = data.pop(0)
        concepts = [row[:-1] for row in data]
        target = [row[-1] for row in data]

    # Initialize S with the first positive example
    specific_h = concepts[0].copy()
    # Initialize G with all '?'
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, instance in enumerate(concepts):
        if target[i] == "Yes":
            # Update General boundary: remove inconsistent hypotheses
            for j in range(len(specific_h)):
                if instance[j] != specific_h[j]:
                    specific_h[j] = '?'
                    for k in range(len(specific_h)):
                        if general_h[k][j] != '?' and general_h[k][j] != specific_h[j]:
                            general_h[k][j] = '?'

        if target[i] == "No":
            # Update General boundary: specialize
            for j in range(len(specific_h)):
                if instance[j] != specific_h[j]:
                    general_h[j][j] = specific_h[j]
                else:
                    general_h[j][j] = '?'

    # Clean up General boundary: remove rows that are only '?'
    indices = [i for i, val in enumerate(general_h) if val == ['?'] * len(specific_h)]
    for i in reversed(indices):
        general_h.pop(i)

    return specific_h, general_h

# Run and Display
s_final, g_final = candidate_elimination('training_data.csv')
print(f"Final Specific Boundary (S): {s_final}")
print(f"Final General Boundary (G): {g_final}")