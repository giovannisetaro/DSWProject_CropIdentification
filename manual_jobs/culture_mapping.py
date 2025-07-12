import pandas as pd
import json

# Load culture file
df_cultures = pd.read_csv('data/REF_CULTURES_GROUPES.csv', sep=';')

# Load mapping rules
with open('data/culture_mapping_rules2.json', 'r', encoding='utf-8') as f:
    mapping_rules = json.load(f)

# Reverse dictionnaire
reversed_dict = {}
for key, values in mapping_rules.items():
    for val in values:
        reversed_dict[val] = key

df_cultures["Label"] = df_cultures["LIBELLE_CULTURE"].map(reversed_dict)

# Save file
output_path = 'data/REF_CULTURES_GROUPES_mapped.csv'
df_cultures.to_csv(output_path, sep=';', index=False)

# Display df for manual check
print(df_cultures.head())
