
### Crop mapping from image time series: deep learning


# Ground truth dataset used to label the dataset: 
- ([online visualisation](https://www.geoportail.gouv.fr/donnees/registre-parcellaire-graphique-rpg-2023))
- ([downloading link](https://data.geopf.fr/telechargement/download/RPG/RPG_2-2__GPKG_LAMB93_FXX_2023-01-01/RPG_2-2__GPKG_LAMB93_FXX_2023-01-01.7z))


# Satelite images used to train the models : 
- [Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A (SR)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED?hl=fr)


We leverages 4 spectral bands from Sentinel-2 satellites to perform crop detection. These bands allow us to extract detailed information about vegetation based on light reflectance at specific wavelengths.

# The following bands are used:

Band 	Resolution	Wavelength (S2A / S2B)	
B2	10 m	496.6 nm / 492.1 nm	Blue	Chlorophyll detection, cloud cover analysis
B3	10 m	560 nm / 559 nm	Green	Vegetation contrast, plant health analysis
B4	10 m	664.5 nm / 665 nm	Red	NDVI calculation, vegetation growth tracking
B8	10 m	835.1 nm / 833 nm	NIR (Near Infrared)	Biomass detection, distinguishing bare soil from vegetation




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
