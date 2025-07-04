
Crop mapping from image time series: deep learning




## biblio 

Turkoglu, M. O., D'Aronco, S., Perich, G., Liebisch, F., Streit, C., Schindler, K., & Wegner, J. D. (2021).  
*Crop mapping from image time series: deep learning with multi-scale label hierarchies*.  
Remote Sensing of Environment, 264. Elsevier.  

[https://doi.org/10.1016/j.rse.2021.112623](https://doi.org/10.1016/j.rse.2021.112623)






Entrée :
X ∈ [B, C=4, T, H, W] — séquence multispectrale temporelle.

Reshape :
vers [B × H × W, C, T] — traitement pixel-wise.

1D CNN temporel multicanal :
convolution sur T, avec C=4 canaux d’entrée → D canaux de sortie.

Reshape inverse :
vers [B, D, H, W] — embedding image.

2D CNN (U-Net ou ResNet).

Classification par pixel.


# DSWProject_CropIdentification