sir_data = [(98, 2, 0), (96, 2, 2), (90, 5, 5)]

# Iterating using index
i_values = []
for i in range(len(sir_data)):
    i_values.append(sir_data[i][1])

print(i_values)

# Iterate over values in lists (or tuples... or arrays.... or pandas series)
s_values = []
for datapoint in sir_data:
    s_values.append(datapoint[0])

# We can also enumerate, if you care about the index
r_values = []
for i, datapoint in enumerate(sir_data):
    r_values.append(datapoint[2])

# This might be overwhelming, so you can ignore it, but check this out
# Combined iteration
for i in range(len(i_values)):
    print((s_values[i], i_values[i], r_values[i]))

# Zip function
for s, i, r in zip(s_values, i_values, r_values):
    print((s, i, r))

# List comprehension
# How do I make a list really easily if it's a simple list?
s_values = [datapoint[0] for datapoint in sir_data]
print(s_values)