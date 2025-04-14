# Evo2 Monosemantic Feature Analysis

This project implements the sparse autoencoder approach described in the paper ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) by Anthropic, but adapted for the Evo2 genomic language model.

## Overview

The sparse autoencoder approach helps decompose complex and polysemantic neurons in neural networks into more interpretable monosemantic features. This can help us better understand what patterns the model has learned to recognize in genomic sequences.

This implementation:

1. Extracts activations from specific layers of the Evo2 model
2. Trains a sparse autoencoder on these activations 
3. Analyzes the resulting features to understand what genomic patterns they represent

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd explaining_evo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure the Evo2 model is properly set up:
```bash
# This will download the Evo2 model if needed
```

## Usage

The main script for running the analysis is `src/explaining/run_evo2_analysis.py`. It can be used to extract activations, train the autoencoder, and analyze the results.

### Basic Usage

```bash
python -m src.explaining.run_evo2_analysis \
  --model_name evo2_7b \
  --layer_name blocks.6.mlp \
  --data_file path/to/your/sequences.fasta \
  --output_dir results \
  --expansion_factor 8.0 \
  --cuda
```

### Command Line Arguments

- `--model_name`: Name of Evo2 model to use ('evo2_7b' or 'evo2_40b')
- `--local_path`: Optional path to local model checkpoint
- `--layer_name`: Name of the layer to extract activations from
- `--data_file`: Path to file containing genomic sequences to analyze
- `--max_samples`: Maximum number of sequences to process
- `--batch_size`: Batch size for processing
- `--expansion_factor`: Expansion factor for autoencoder (latent_dim = input_dim * expansion_factor)
- `--l1_coefficient`: L1 sparsity coefficient for the autoencoder
- `--epochs`: Number of training epochs
- `--lr`: Learning rate for training
- `--tied_weights`: Use tied weights in the autoencoder
- `--output_dir`: Directory to save results
- `--seed`: Random seed for reproducibility
- `--save_activations`: Save raw activations (warning: can be large)
- `--cuda`: Use CUDA if available
- `--verbose`: Print verbose output
- `--mode`: Action to perform: extract activations, train autoencoder, analyze features, or all

### Example: Extracting Activations Only

```bash
python -m src.explaining.run_evo2_analysis \
  --model_name evo2_7b \
  --layer_name blocks.6.mlp \
  --data_file path/to/your/sequences.fasta \
  --output_dir results \
  --mode extract \
  --save_activations \
  --cuda
```

### Example: Training on Pre-extracted Activations

```bash
python -m src.explaining.run_evo2_analysis \
  --model_name evo2_7b \
  --layer_name blocks.6.mlp \
  --data_file path/to/your/sequences.fasta \
  --output_dir results \
  --mode train \
  --expansion_factor 8.0 \
  --l1_coefficient 1e-3 \
  --epochs 50 \
  --cuda
```

### Example: Analyzing Pre-trained Features

```bash
python -m src.explaining.run_evo2_analysis \
  --model_name evo2_7b \
  --layer_name blocks.6.mlp \
  --data_file path/to/your/sequences.fasta \
  --output_dir results \
  --mode analyze \
  --expansion_factor 8.0 \
  --cuda
```

## Interpreting the Results

After running the full analysis pipeline, you'll find several outputs in the results directory:

1. **Activations**: Raw activations extracted from the model (if `--save_activations` was used)
2. **Models**: Trained sparse autoencoder checkpoints 
3. **Analysis**: Feature analysis results including:
   - Feature density histograms
   - Feature clusters visualizations
   - Individual feature activation patterns
   - Feature weight patterns

## Project Structure

- `src/explaining/hooks/`: Hooks for extracting activations from the model
- `src/explaining/evo2_activations.py`: Utilities for extracting activations from Evo2
- `src/explaining/sparse_autoencoder.py`: Implementation of the sparse autoencoder
- `src/explaining/feature_analysis.py`: Tools for analyzing learned features
- `src/explaining/run_evo2_analysis.py`: Main script for running the analysis pipeline

## Next Steps

After running the analysis, consider these next steps:

1. Look for biologically meaningful features in your analysis results
2. Compare features across different layers of the model
3. Try different expansion factors to see how features split and combine
4. Use the feature activations to steer the model's generation or attention
5. Manually annotate the most interesting features and build a dictionary of genomic motifs

## References

- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Evo2: Evolution at Scale](https://github.com/ArcInstitute/evo2)
# explaining_evo
