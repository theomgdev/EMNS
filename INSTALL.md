# EMNS Installation and Quick Start Guide

## Installation

### Option 1: Direct Installation
```bash
# Clone the repository
git clone <repository-url>
cd EMNS

# Install dependencies
pip install -r requirements.txt

# Run the main demo
python lib_and_demo.py
```

### Option 2: Install as Package
```bash
# Clone the repository
git clone <repository-url>
cd EMNS

# Install as a package
pip install .

# Or install in development mode
pip install -e .
```

## Quick Start

### Basic Usage
```python
from lib_and_demo import EMNSNetwork, create_test_data

# Create test data
X_train, y_train = create_test_data(n_samples=100, n_features=2)

# Create EMNS network
network = EMNSNetwork(layer_sizes=[2, 8, 4, 1], mutation_rate=0.3)

# Train the network
history = network.train(X_train, y_train, epochs=200)

# Make predictions
predictions = network.predict(X_train[:5])
```

### Running Examples
```bash
# Run the main demonstration
python lib_and_demo.py

# Run additional examples
python example_usage.py
```

## Key Features

- **No Gradient Computation**: Uses evolutionary principles instead of backpropagation
- **Self-Organizing**: Parameters develop resistance to protect important configurations
- **Modular Architecture**: Emerges naturally through resistance clustering
- **Continual Learning**: Can learn new tasks without forgetting previous ones
- **Hardware Efficient**: Linear complexity and highly parallelizable

## File Structure

- `lib_and_demo.py`: Main EMNS implementation and demonstration
- `example_usage.py`: Additional usage examples
- `requirements.txt`: Python dependencies
- `setup.py`: Package setup file
- `README.md`: Detailed technical documentation
- `INSTALL.md`: This installation guide

## System Requirements

- Python 3.7+
- NumPy 1.21.0+
- Matplotlib 3.5.0+

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install --upgrade numpy matplotlib
```

### Visualization Issues
If matplotlib plots don't show, try:
```bash
# On macOS with conda
conda install python.app

# Or use non-interactive backend
export MPLBACKEND=Agg
```

### Performance Issues
- Reduce network size for faster training
- Lower mutation rate for more stable convergence
- Use fewer epochs for quick testing

## Next Steps

1. Read the detailed README.md for theoretical background
2. Experiment with different network architectures
3. Try the continual learning examples
4. Modify mutation parameters for your specific use case

For questions or issues, refer to the README.md or contact the author. 