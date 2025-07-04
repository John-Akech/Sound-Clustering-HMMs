# Sound Clustering Analysis

## Project Overview

This project implements unsupervised machine learning techniques to cluster unlabeled sound data. Using Mel spectrogram features, dimensionality reduction, and multiple clustering algorithms, we discover hidden patterns and group similar audio files together.

## Features

- **Audio Feature Extraction**: Mel spectrogram feature extraction from .wav files
- **Dimensionality Reduction**: PCA and t-SNE for visualization and analysis
- **Clustering Algorithms**: K-Means and DBSCAN implementation
- **Optimization**: Automated cluster number optimization using silhouette analysis
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Detailed Analysis**: Feature interpretation and cluster characteristics

## Requirements

```
numpy
pandas
matplotlib
seaborn
librosa
scikit-learn
```

## Installation

```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn
```

## Usage

1. **Setup Data Path**: Update the `data_path` variable in the notebook to point to your audio files directory
2. **Run Notebook**: Execute all cells in `sound_clustering.ipynb`
3. **View Results**: Check generated visualizations and `clustering_results.csv`

## Project Structure

```
ML_Technique2/
├── Dataset/
│   └── unlabelled_sounds/   # Audio files directory
├── sound_clustering.ipynb   # Main analysis notebook
├── clustering_results.csv   # Clustering results
├── requirements.txt         # Python dependencies
├── README.md                # This file (README.md)
└── HMM for Flood Risk Prediction in South Sudan.pdf  # PDF Document for Part 2
```

## Methodology

### 1. Feature Extraction
- Load audio files (.wav format)
- Extract 13-dimensional Mel spectrogram features
- Aggregate features using mean across time axis

### 2. Dimensionality Reduction
- **PCA**: Linear dimensionality reduction preserving global variance
- **t-SNE**: Non-linear reduction preserving local neighborhood structure
- Compare both methods for visualization effectiveness

### 3. Clustering
- **K-Means**: Partition-based clustering with optimized cluster count
- **DBSCAN**: Density-based clustering with noise detection
- Evaluate using silhouette score and Davies-Bouldin index

### 4. Evaluation
- Quantitative metrics (silhouette score, Davies-Bouldin index)
- Visual analysis using 2D/3D projections
- Cluster characteristic analysis

## Key Results

- **Feature Extraction**: Successfully captured audio characteristics using Mel spectrograms
- **Dimensionality Reduction**: t-SNE provided superior cluster visualization compared to PCA
- **Clustering Performance**: K-Means showed consistent performance with optimized parameters
- **Algorithm Comparison**: Different algorithms revealed complementary data structure aspects

## Visualizations

The notebook generates several key visualizations:

1. **Raw Feature Plots**: Initial 2D scatter plots showing high-dimensional data limitations
2. **3D Dimensionality Reduction**: PCA vs t-SNE comparison in 3D space
3. **Optimization Plots**: Elbow method and silhouette analysis for optimal K
4. **Clustering Results**: 2x2 grid comparing algorithms on different projections
5. **Feature Analysis**: Cluster characteristics and distributions

## Output Files

- `clustering_results.csv`: Contains filename, K-Means cluster, and DBSCAN cluster assignments
- Various plots and visualizations displayed in notebook

## Technical Notes

### Audio Processing
- Sample rate: 22,050 Hz
- Duration: First 3 seconds of each file
- Feature type: Mel spectrogram (13 bands)
- Preprocessing: Standardization (mean=0, std=1)

### Clustering Parameters
- K-Means: Optimized K using silhouette analysis
- DBSCAN: eps=0.5, min_samples=5
- Random state: 42 for reproducibility

## Interpretation Guidelines

### Silhouette Score
- Range: -1 to 1
- Higher values indicate better cluster separation
- Values > 0.5 suggest reasonable clustering

### Davies-Bouldin Index
- Range: 0 to infinity
- Lower values indicate better clustering
- Measures average similarity between clusters

## Future Improvements

1. **Feature Enhancement**:
   - Add MFCC features
   - Include spectral features (centroid, rolloff, etc.)
   - Experiment with different time aggregation methods

2. **Algorithm Expansion**:
   - Gaussian Mixture Models
   - Hierarchical clustering
   - Ensemble clustering methods

3. **Evaluation Enhancement**:
   - Cross-validation techniques
   - Stability analysis
   - External validation metrics

## Troubleshooting

### Common Issues

1. **File Path Error**: Ensure the `data_path` variable points to correct directory
2. **Missing Dependencies**: Install all required packages using pip
3. **Audio Format**: Ensure audio files are in .wav format
4. **Memory Issues**: Reduce `max_files` parameter for large datasets

### Performance Tips

- For large datasets, consider processing in batches
- Use parallel processing for feature extraction
- Optimize t-SNE parameters (perplexity, learning_rate) for your data size

## License

This project is for educational purposes. Feel free to modify and extend for your own analysis.

## Contact

For questions or improvements, please refer to the detailed comments and markdown cells within the notebook.
