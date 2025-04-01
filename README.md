## Hybrid CNN + MLP Model for Wildfire Risk Classification

This repository contains the full implementation of a hybrid deep learning architecture designed to classify wildfire risk levels by integrating satellite images and environmental data. The project was conducted as part of the AI4EO final assignment.

All image files used in this study (for training, validation, and test sets) are sourced from the public FireRisk dataset.
The official repository can be found here: https://github.com/CharmonyShen/FireRisk

The original dataset includes only train and validation images. To enable model evaluation, a test set was constructed by randomly sampling 10% from the training set, stratified by class.

**Note:** Image files are not included in this repository due to size and licensing. Users should download the dataset from the original source and replicate the structure.

### Final Class Distribution

**Training Set:**
- High: 5667 images  
- Very High: 1385 images  
- Moderate: 7752 images  
- Low: 9582 images  
- Very Low: 15255 images  
- Non-burnable: 11614 images  
- Water: 1395 images

**Validation Set:**
- High: 1308 images  
- Very High: 318 images  
- Moderate: 1804 images  
- Low: 1795 images  
- Very Low: 1805 images  
- Non-burnable: 1795 images  
- Water: 180 images

**Test Set (sampled from training set):**
- High: 629 images  
- Very High: 326 images  
- Moderate: 861 images  
- Low: 1070 images  
- Very Low: 1695 images  
- Non-burnable: 1795 images  
- Water: 172 images

The region of interest covers the western United States, approximately spanning longitude -125 to -100 and latitude 40 to 45, based on coordinates embedded in image filenames from the FireRisk dataset. This area includes fire-prone zones such as California, Oregon, and Colorado.

A 15-year period (2009–2024) was selected to align MODIS vegetation indices, historical fire activity, and ERA5 climate variables. This time window was defined independently of FireRisk, ensuring temporal consistency across all external sources used for tabular features.

---

## Data Preprocessing and Feature Engineering

To construct the final tabular feature set, spatial and temporal alignment was performed across multiple sources. The processed features include:

### MODIS NDVI and NBR Matching

MODIS-based NDVI and NBR values (2009–2024) were matched to FireRisk images based on latitude, longitude, and acquisition date (parsed from filenames).

**Code Snippets:**
- `Combining csv data from 4 different regions downloaded with GEE`
- `Demonstrates how MODIS NDVI/NBR values were matched to image coordinates and dates`

GEE Extraction script is included under:
```
preprocessing/gee_ndvi_nbr_download.txt
```

### Fire History (FIRMS)
NASA FIRMS variables (brightness, FRP, confidence) were integrated using a ±3-day matching window to account for detection uncertainty and time differences.

**Code Snippet:**
- `Sample Snippet: Matching FIRMS Fire Data`

### Climate Features (ERA5)
ERA5 hourly climate data (temperature, wind speed, precipitation) were aggregated at each image’s spatio-temporal point and included as environmental predictors.

**Code Snippets:**
- `Sample Snippet of Climate Processing`
- `ERA5-Flattened Tabular Matching`

### Domain-Inspired Features
Several interpretable features were computed using combinations of the above data:

- `burn_count`: total fire detections over time at the same location
- `fire_potential`: calculated as `temp_c × precip_m`
- `wind_conf`: combined `frp`, `wind_mps`, and `confidence`
- `risk_factor`: a product of `burned_veg_loss` and `high_heat_stress`
- `risk_persistence`: `burn_count × fire_potential`, indicating historical fire severity

These features are implemented under:
- `Domain-Inspired Features`
- `Creating final CSVs by bringing all the features together`

All final features were saved in `.csv` format and matched with corresponding image filenames using coordinate-time keys.

### Output CSVs
Final merged CSVs used for model input are located in the `/data` folder:
```
/data/train_Final.csv
/data/val_Final.csv
/data/test_Final.csv
```
##  Model Architecture

This project implements a **hybrid deep learning model** that combines satellite images and tabular environmental features for wildfire risk classification.

The architecture is composed of:

- **CNN Backbone (ResNet-18):** Extracts visual features from FireRisk RGB images (224×224).
- **MLP Branch:** Processes tabular data including NDVI/NBR, ERA5 climate, and FIRMS fire features.
- **Fusion Layer:** Image and tabular embeddings are concatenated and passed through fully connected layers for classification.

---

## Training Configuration

| Parameter         | Value                                  |
|------------------|----------------------------------------|
| CNN Backbone     | ResNet-18 (pretrained)                 |
| Input Size       | 224 × 224                              |
| Tabular Features | NDVI, NBR, ERA5, FIRMS, Domain Features|
| Fusion Method    | Concatenation                          |
| Optimizer        | Adam (lr = 1e-4)                       |
| Scheduler        | Cosine Annealing                       |
| Epochs           | 7                                     |
| Batch Size       | 32                                     |
| Loss Function    | CrossEntropyLoss                       |
| Early Stopping   | Based on Macro F1                      |

---

## Evaluation Metrics

The model was evaluated using:

-  **Accuracy**
-  **Macro F1-Score**
-  **Per-class Precision, Recall, F1**
-  **Confusion Matrix**

Evaluation was conducted on a hold-out test set (10% of the training data, stratified by class).

---

## Pretrained Checkpoints

Pre-trained model weights from the hybrid architecture (CNN + MLP) are available:

| File Name              | Description                         | Download Link |
|------------------------|-------------------------------------|---------------|
| `checkpoint_final.pth` | Final epoch checkpoint              | [📥 Download](https://drive.google.com/file/d/1yVtB89RRlyWMx6I4v0eVRQ4T-Ry_WxjJ/view?usp=sharing) |
| `best_model_final.pth` | Best model (highest val F1-score)   | [📥 Download](https://drive.google.com/file/d/1yVtB89RRlyWMx6I4v0eVRQ4T-Ry_WxjJ/view?usp=sharing) |

>  Make sure to place `.pth` files in the working directory or update the `load_model()` function accordingly.


---

## Requirements

To run this project, make sure the following Python packages are installed:

torch>=1.12  
torchvision  
numpy  
pandas  
scikit-learn  
matplotlib  
seaborn  
opencv-python  
tqdm   

---

##  Usage (Training)

All training steps are handled inside the notebook `FireRisk_Model.ipynb`.

Make sure the following inputs are correctly prepared:

-  `train_Final.csv` and `val_Final.csv` (tabular features)
-  FireRisk images organized into `/train` and `/val` directories, each with 7 class folders:
  - Very_Low, Low, Moderate, High, Very_High, Non-burnable, Water

Once everything is ready, launch the notebook and run all cells.  
Training will proceed with early stopping and model checkpointing enabled.  
The best performing model (on validation set) will be saved as:
best_model_final.pth


##  Results (Test Set Performance)

The model was evaluated on a hold-out test set (10% of training data, stratified by class).

###  Overall Metrics:

| Metric           | Score  |
|------------------|--------|
| Accuracy         | 0.853  |
| Macro F1-Score   | 0.821  |

###  Per-Class F1 Scores:

| Class          | F1-Score |
|----------------|----------|
| Very Low       | 0.89     |
| Low            | 0.85     |
| Moderate       | 0.81     |
| High           | 0.78     |
| Very High      | 0.72     |
| Non-burnable   | 0.84     |
| Water          | 0.88     |

###  Confusion Matrix

A visual confusion matrix is included in the notebook for deeper analysis.  
It shows clear confusion between `High` and `Very High` classes, while classes like `Water` and `Very Low` are easily distinguished.
