# E2E Constrain Project

## Overview
This project implements an end-to-end weather forecasting system with constraints, combining multiple components for weather data assimilation, forecasting, and downscaling. The system utilizes deep learning models to process and predict weather patterns with high accuracy.

## Project Structure
```
.
├── code/                    # Main implementation code
│   ├── canglong/           # Core model implementations
│   ├── constant_masks/     # Mask definitions for data processing
│   ├── data/              # Data handling utilities
│   ├── utils/             # Utility functions
│   └── *.ipynb            # Jupyter notebooks for testing and analysis
├── doc/                    # Documentation
└── repo/                   # External repository dependencies
```

## Key Components

### 1. Core Models
- **Conv4D**: Implementation of 4D convolutional neural networks for spatiotemporal data processing
- **Recovery**: Data recovery and reconstruction modules
- **Embed**: Embedding layers for feature representation

### 2. Training Pipeline
- End-to-end training system with distributed data parallel (DDP) support
- Multiple loss functions including:
  - Weighted RMSE
  - Pressure-weighted RMSE
  - Standard RMSE
  - Downscaling RMSE

### 3. Data Processing
- Support for ERA5 weather data
- Custom data loaders for training and validation
- Data assimilation and forecasting capabilities

## Features
- Distributed training support using PyTorch DDP
- Multiple model architectures for different weather prediction tasks
- Flexible loss function selection
- Support for various weather variables (temperature, wind, pressure, etc.)
- Custom 4D convolutional operations for spatiotemporal data

## Usage

### Training
The system supports multiple training modes:
1. Assimilation training
2. Forecast training
3. End-to-end training
4. Fine-tuning

Example training command:
```bash
python e2e_train.py \
    --output_dir <output_directory> \
    --loss lw_rmse \
    --batch_size 3 \
    --epoch 10 \
    --lead_time <time_steps> \
    --era5_mode 4u
```

### Model Configuration
Key parameters include:
- `lead_time`: Number of time steps to forecast
- `era5_mode`: ERA5 data processing mode
- `res`: Resolution setting
- `frequency`: Data sampling frequency
- `region`: Target region for prediction

## Dependencies
- PyTorch
- NumPy
- CUDA support for GPU acceleration
- Additional dependencies as specified in the notebooks

## Documentation
Detailed documentation and usage examples can be found in the Jupyter notebooks:
- `how_to_run.ipynb`: Basic usage instructions
- `generate_weekly.ipynb`: Weekly data generation process
- `test.ipynb`: Testing and validation procedures

## License
[Specify license information]

## Contributing
[Specify contribution guidelines] 