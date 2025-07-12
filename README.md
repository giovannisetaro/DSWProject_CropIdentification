
# Crop mapping from multispectral image time series: deep learning

## Dataset Description

### Ground Truth (Crop Labels) 

We use the Registre Parcellaire Graphique (RPG) 2023 as the reference for crop types and parcel boundaries in France.

- ([online visualisation](https://www.geoportail.gouv.fr/donnees/registre-parcellaire-graphique-rpg-2023))
- ([downloading link](https://data.geopf.fr/telechargement/download/RPG/RPG_2-2__GPKG_LAMB93_FXX_2023-01-01/RPG_2-2__GPKG_LAMB93_FXX_2023-01-01.7z))

# This table lists key agricultural zones across France, paired with nearby towns for geographic reference. 
# Each zone represents distinct agricultural landscapes characterized by dominant crops and farming practices due to various pedoclimatic context.


| Zone                            | Region                 | Nearby Town (Map Reference) | Notes                                                         |
| ------------------------------- | ---------------------- | --------------------------- | ------------------------------------------------------------- |
| **Nord-Picardie**               | Hauts-de-France        | **Saint-Quentin** (Aisne)   | Surrounded by large-scale crops (wheat, sugar beet, potatoes) |
| **Paris Basin**                 | Île-de-France / Centre | **Chartres** (Eure-et-Loir) | Heart of the Beauce, vast cereal plains                       |
| **Brittany / Pays de la Loire** | Brittany / Vendée      | **Vitré** (Ille-et-Vilaine) | Mixed zone: livestock, silage maize, hedgerows                |
| **Southwest**                   | Nouvelle-Aquitaine     | **Auch** (Gers)             | Cereal polyculture, maize, sunflower                          |
| **Southeast**                   | Provence, Rhône-Alpes  | **Carpentras** (Vaucluse)   | Vineyards, orchards, greenhouse vegetable farming             |
| **Massif Central**              | Auvergne               | **Riom** (Puy-de-Dôme)      | Limagne plain: polyculture on volcanic plains                 |
| **Alsace  / Lorraine**           | Grand Est              | **Colmar** (Haut-Rhin)      | Hillside vineyards + lowland crop farming                     |
| **Mediterranean**               | Occitanie, PACA        | **Béziers** (Hérault)       | Vineyards, olive trees, vegetable crops, dry climate          |

We use ESA WorldCover (10 m resolution, global) to identify highly cultivated areas for sampling. Class 40 in this dataset represents "Cropland," allowing precise detection of agricultural zones in 2020 and 2021. we select 5 zones around each of those representatives landscapes

### Satellite Data
We rely on Sentinel-2 Level-2A (Surface Reflectance) images: [Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A (SR)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED?hl=fr)


We leverages 4 spectral bands from Sentinel-2 satellites to perform crop detection. These bands allow us to extract detailed information about vegetation based on light reflectance at specific wavelengths.

### The following bands are used:

| Band | Name  | Resolution | Wavelength (S2A / S2B) | Description                                         |
| ---- | ----- | ---------- | ---------------------- | --------------------------------------------------- |
| B2   | Blue  | 10 m       | 496.6 / 492.1 nm       | Chlorophyll detection, cloud cover analysis         |
| B3   | Green | 10 m       | 560 / 559 nm           | Vegetation contrast, plant health analysis          |
| B4   | Red   | 10 m       | 664.5 / 665 nm         | NDVI calculation, vegetation growth tracking        |
| B8   | NIR   | 10 m       | 835.1 / 833 nm         | Biomass detection, distinguishes soil vs vegetation |

![Demo du projet](.plots/gif/zone3_B2.gif)

# Models 

## Deep Learning Model Overview : 
Our model is designed to classify each pixel of a multispectral time series image into crop types. It combines temporal and spatial feature extraction using a hybrid CNN architecture.

Input : X ∈ [B, C=4, T, H, W]
A batch of multispectral sequences with 4 channels (e.g. Red, Green, Blue, NIR), T time steps, and spatial size H×W.

Step 1: Temporal Encoding (Pixel-wise)

Reshape to [B × H × W, C, T]

Apply a 1D CNN over time (T) independently for each pixel.
→ Captures temporal patterns per pixel across spectral bands.
→ Outputs D-dimensional embeddings.

Step 2: Reshape to Spatial Grid

Reshape back to [B, D, H, W]
→ Builds a pseudo-image from temporal embeddings.

Step 3: Spatial Encoding (Image-wise)

Pass through a 2D CNN backbone (here it's a simplify version of a U-Net).
→ Captures spatial context and structure between neighboring pixels.

Step 4: Classification

Output is per-pixel class logits: shape [B, num_classes, H, W]


## biblio 

Turkoglu, M. O., D'Aronco, S., Perich, G., Liebisch, F., Streit, C., Schindler, K., & Wegner, J. D. (2021).  
*Crop mapping from image time series: deep learning with multi-scale label hierarchies*.  
Remote Sensing of Environment, 264. Elsevier.  
