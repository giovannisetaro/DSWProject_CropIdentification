import pandas as pd

# Chargement du fichier original
df = pd.read_csv('data/REF_CULTURES_GROUPES.csv', sep=';')

# Liste de référence des cultures
cultures = [
    "Meadow", "Winter wheat", "Maize", "Pasture", "Sugar beet", "Winter barley",
    "Winter rapeseed", "Vegetables", "Potatoes", "Wheat", "Sunflowers", "Vines",
    "Spelt", "Hedge", "Apples", "Soy", "Fallow", "Peas", "Berries", "Oat",
    "Field bean", "Einkorn wheat", "Rye", "Tree crop", "Summer wheat", "Stone fruit",
    "Mixed crop", "Pears", "Chicory", "Sorghum", "Forest", "Summer barley",
    "Legumes", "Grain", "Biodiversity area", "Linen", "Tobacco", "Pumpkin",
    "Hemp", "Buckwheat", "Summer rapeseed", "Hops", "Multiple", "Beets", "Lupine",
    "Mustards", "Gardens", "Chestnut"
]

# Fonction de mapping sémantique
def map_label(name):
    s = name.lower()
    if 'prairie' in s:
        return 0
    if 'blé tendre' in s and 'hiver' in s:
        return 1
    if 'blé tendre' in s and 'printemps' in s:
        return 24
    if 'blé dur' in s or ('blé' in s and 'tendre' not in s):
        return 9
    if 'maïs' in s or 'zea' in s:
        return 2
    if 'orge' in s and 'hiver' in s:
        return 5
    if 'orge' in s and 'printemps' in s:
        return 31
    if 'avoine' in s:
        return 19
    if 'seigle' in s:
        return 22
    if 'sorgho' in s:
        return 29
    if 'millet' in s:
        return 33  # Grain
    if 'betterave sucr' in s:
        return 4
    if 'betterave' in s:
        return 43
    if 'colza' in s or 'brassica napus' in s:
        if 'hiver' in s:
            return 6
        if 'printemps' in s or 'été' in s:
            return 40
        return 6
    if 'tournesol' in s or 'helianthus' in s:
        return 10
    if 'épeautre' in s:
        return 12
    if 'pois' in s:
        return 17
    if 'soja' in s:
        return 15
    if 'arachide' in s or 'arachis' in s:
        return 32
    if 'lupine' in s:
        return 44
    if 'sarrasin' in s or 'fagopyrum' in s:
        return 39
    if 'lin ' in s or 'fibres' in s:
        return 35
    if 'courge' in s or 'potiron' in s:
        return 37
    if 'moutarde' in s or 'brassica' in s:
        return 45
    if 'mélange' in s:
        return 26
    if 'houblon' in s or 'hop' in s:
        return 41
    if any(x in s for x in ['légume', 'potager']):
        return 7
    if 'pomme de terre' in s or 'pommes de terre' in s:
        return 8
    if 'vigne' in s:
        return 11
    if 'bois' in s:
        return 30
    if 'pomme' in s:
        return 14
    if 'poir' in s:
        return 27
    if 'cerise' in s or 'abricot' in s or 'pêch' in s:
        return 25
    if 'fraise' in s or 'framboise' in s or 'mûre' in s:
        return 18
    if 'arbre' in s:
        return 23
    if 'jachère' in s:
        return 16
    if 'haie' in s:
        return 13
    if 'biodiversité' in s:
        return 34
    if 'jardin' in s:
        return 46
    return None

# Application du mapping
df['Label'] = df['LIBELLE_CULTURE'].apply(map_label)

# Sauvegarde du nouveau fichier
output_path = 'data/REF_CULTURES_GROUPES_mapped.csv'
df.to_csv(output_path, sep=';', index=False)

# Affichage d'aperçu
df.head()
