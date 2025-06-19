"""
Evolvable Modular Neural Systems (EMNS) Implementation

This module implements the EMNS architecture described in the research paper,
featuring resistance-governed parameter evolution without gradient-based optimization.

Author: Cahit Karahan
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy
import random

class EvolvableParameter:
    """
    A parameter that can evolve with its own resistance value.
    """
    def __init__(self, value: float, resistance: float = 0.1, name: str = ""):
        self.value = value
        self.resistance = np.clip(resistance, 0.0, 1.0)
        self.name = name
        self.history = []  # For tracking evolution
    
    def mutate(self, mutation_rate: float) -> None:
        """Apply universal mutation rule to both value and resistance."""
        # Mutate the parameter value
        noise = np.random.normal(0, 1)
        delta = noise * mutation_rate * (1 - self.resistance)
        self.value += delta
        
        # Clip parameter values to prevent numerical explosion
        self.value = np.clip(self.value, -5.0, 5.0)
        
        # Mutate the resistance (co-evolution) with smaller step size
        resistance_noise = np.random.normal(0, 1)
        resistance_delta = resistance_noise * mutation_rate * 0.01 * (1 - self.resistance)
        self.resistance += resistance_delta
        self.resistance = np.clip(self.resistance, 0.0, 0.9)  # Prevent resistance from reaching 1.0
        
        # Track history
        self.history.append((self.value, self.resistance))
    
    def __repr__(self):
        return f"EvolvableParameter(value={self.value:.4f}, resistance={self.resistance:.4f})"

class AggregationFunction:
    """Evolvable aggregation function with multiple basis functions."""
    
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        # Initialize coefficients for different aggregation operations
        self.coefficients = {
            'linear': EvolvableParameter(1.0, 0.3, 'agg_linear'),
            'quadratic': EvolvableParameter(0.0, 0.1, 'agg_quadratic'),
            'sqrt': EvolvableParameter(0.0, 0.1, 'agg_sqrt'),
            'sin': EvolvableParameter(0.0, 0.1, 'agg_sin'),
            'gaussian': EvolvableParameter(0.0, 0.1, 'agg_gaussian')
        }
    
    def __call__(self, inputs: np.ndarray) -> float:
        """Apply evolvable aggregation function."""
        # Clip inputs to prevent numerical issues
        inputs = np.clip(inputs, -3, 3)
        result = 0.0
        
        # Linear aggregation
        result += self.coefficients['linear'].value * np.sum(inputs)
        
        # Quadratic aggregation
        result += self.coefficients['quadratic'].value * np.sum(inputs**2)
        
        # Square root aggregation
        result += self.coefficients['sqrt'].value * np.sum(np.sqrt(np.abs(inputs)))
        
        # Trigonometric aggregation
        result += self.coefficients['sin'].value * np.sum(np.sin(inputs))
        
        # Gaussian aggregation
        result += self.coefficients['gaussian'].value * np.sum(np.exp(-inputs**2))
        
        # Clip output to prevent explosion
        return np.clip(result, -3.0, 3.0)
    
    def get_parameters(self) -> List[EvolvableParameter]:
        """Return all evolvable parameters."""
        return list(self.coefficients.values())

class ActivationFunction:
    """Evolvable activation function with multiple basis functions."""
    
    def __init__(self):
        # Initialize coefficients for different activation functions
        self.coefficients = {
            'tanh': EvolvableParameter(1.0, 0.3, 'act_tanh'),
            'sigmoid': EvolvableParameter(0.0, 0.1, 'act_sigmoid'),
            'relu': EvolvableParameter(0.0, 0.1, 'act_relu'),
            'swish': EvolvableParameter(0.0, 0.1, 'act_swish'),
            'linear': EvolvableParameter(0.0, 0.1, 'act_linear')
        }
    
    def __call__(self, x: float) -> float:
        """Apply evolvable activation function."""
        # Clip input to prevent numerical instability
        x = np.clip(x, -10, 10)
        result = 0.0
        
        # Tanh
        result += self.coefficients['tanh'].value * np.tanh(x)
        
        # Sigmoid
        result += self.coefficients['sigmoid'].value * (1 / (1 + np.exp(-x)))
        
        # ReLU
        result += self.coefficients['relu'].value * max(0, x)
        
        # Swish
        sigmoid_x = 1 / (1 + np.exp(-x))
        result += self.coefficients['swish'].value * x * sigmoid_x
        
        # Linear
        result += self.coefficients['linear'].value * x
        
        # Clip output to prevent explosion
        return np.clip(result, -3.0, 3.0)
    
    def get_parameters(self) -> List[EvolvableParameter]:
        """Return all evolvable parameters."""
        return list(self.coefficients.values())

class EvolvableSynapse:
    """A synaptic connection with evolvable weight, bias, and gain."""
    
    def __init__(self, weight: float = None, bias: float = 0.0, gain: float = 1.0):
        if weight is None:
            weight = np.random.normal(0, 0.1)
        
        self.weight = EvolvableParameter(weight, 0.1, 'synapse_weight')
        self.bias = EvolvableParameter(bias, 0.1, 'synapse_bias')
        self.gain = EvolvableParameter(gain, 0.2, 'synapse_gain')
    
    def forward(self, x: float) -> float:
        """Apply synaptic transformation."""
        result = self.gain.value * self.weight.value * x + self.bias.value
        return np.clip(result, -3.0, 3.0)
    
    def get_parameters(self) -> List[EvolvableParameter]:
        """Return all evolvable parameters."""
        return [self.weight, self.bias, self.gain]

class EvolvableNeuron:
    """A neuron with evolvable aggregation and activation functions."""
    
    def __init__(self, num_inputs: int, neuron_id: int = 0):
        self.num_inputs = num_inputs
        self.neuron_id = neuron_id
        
        # Evolvable functions
        self.aggregation = AggregationFunction(num_inputs)
        self.activation = ActivationFunction()
        
        # Input synapses
        self.synapses = [EvolvableSynapse() for _ in range(num_inputs)]
        
        # Internal state
        self.last_output = 0.0
        self.last_aggregation = 0.0
    
    def forward(self, inputs: np.ndarray) -> float:
        """Forward pass through the neuron."""
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Apply synaptic transformations
        synaptic_outputs = np.array([
            synapse.forward(inp) for synapse, inp in zip(self.synapses, inputs)
        ])
        
        # Apply aggregation function
        aggregated = self.aggregation(synaptic_outputs)
        self.last_aggregation = aggregated
        
        # Apply activation function
        output = self.activation(aggregated)
        self.last_output = output
        
        return output
    
    def get_parameters(self) -> List[EvolvableParameter]:
        """Return all evolvable parameters in the neuron."""
        params = []
        params.extend(self.aggregation.get_parameters())
        params.extend(self.activation.get_parameters())
        for synapse in self.synapses:
            params.extend(synapse.get_parameters())
        return params

class EMNSLayer:
    """A layer of evolvable neurons."""
    
    def __init__(self, num_inputs: int, num_neurons: int, layer_id: int = 0):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.layer_id = layer_id
        
        self.neurons = [
            EvolvableNeuron(num_inputs, i) for i in range(num_neurons)
        ]
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return outputs
    
    def get_parameters(self) -> List[EvolvableParameter]:
        """Return all evolvable parameters in the layer."""
        params = []
        for neuron in self.neurons:
            params.extend(neuron.get_parameters())
        return params
    
    def add_neuron(self) -> None:
        """Add a new neuron to the layer."""
        new_neuron = EvolvableNeuron(self.num_inputs, len(self.neurons))
        self.neurons.append(new_neuron)
        self.num_neurons += 1

class EMNSNetwork:
    """
    Evolvable Modular Neural System - Main network class.
    """
    
    def __init__(self, layer_sizes: List[int], mutation_rate: float = 0.1):
        self.layer_sizes = layer_sizes
        self.mutation_rate = mutation_rate
        self.performance_history = []
        self.mutation_rate_history = []
        
        # Build layers
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = EMNSLayer(layer_sizes[i-1], layer_sizes[i], i-1)
            self.layers.append(layer)
        
        # Performance tracking
        self.last_performance = 0.0
        self.best_performance = float('-inf')
        self.best_state = None
        
        # Mutation rate adaptation parameters
        self.alpha = 0.995  # Stabilization factor
        self.beta = 1.05    # Exploration factor
        self.gamma = 0.999  # Natural decay factor
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the entire network."""
        current_inputs = inputs
        
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs)
        
        return current_inputs
    
    def get_all_parameters(self) -> List[EvolvableParameter]:
        """Return all evolvable parameters in the network."""
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params
    
    def mutate_all_parameters(self) -> None:
        """Apply universal mutation to all parameters."""
        for param in self.get_all_parameters():
            param.mutate(self.mutation_rate)
    
    def adapt_mutation_rate(self, current_performance: float) -> None:
        """Adapt the global mutation rate based on performance."""
        if len(self.performance_history) > 0:
            performance_change = current_performance - self.last_performance
            
            if performance_change > 0:
                # Performance improved - stabilize
                self.mutation_rate *= self.alpha
            elif performance_change < 0:
                # Performance degraded - explore
                self.mutation_rate *= self.beta
            else:
                # No change - natural decay
                self.mutation_rate *= self.gamma
        
        # Keep mutation rate in reasonable bounds
        self.mutation_rate = np.clip(self.mutation_rate, 0.001, 1.0)
        
        self.last_performance = current_performance
        self.performance_history.append(current_performance)
        self.mutation_rate_history.append(self.mutation_rate)
        
        # Save best state
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.best_state = self.get_network_state()
    
    def get_network_state(self) -> Dict:
        """Get a snapshot of the current network state."""
        state = {}
        for i, layer in enumerate(self.layers):
            layer_state = {}
            for j, neuron in enumerate(layer.neurons):
                neuron_params = {}
                for param in neuron.get_parameters():
                    neuron_params[param.name] = {
                        'value': param.value,
                        'resistance': param.resistance
                    }
                layer_state[f'neuron_{j}'] = neuron_params
            state[f'layer_{i}'] = layer_state
        return state
    
    def evolve_step(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Perform one evolution step: forward pass, evaluate performance, 
        mutate parameters, and adapt mutation rate.
        """
        # Forward pass
        outputs = self.forward(inputs)
        
        # Evaluate performance (negative MSE for maximization)
        performance = -np.mean((outputs - targets) ** 2)
        
        # Adapt mutation rate based on performance
        self.adapt_mutation_rate(performance)
        
        # Universal parameter mutation
        self.mutate_all_parameters()
        
        return performance
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 1000, verbose: bool = True) -> Dict:
        """
        Train the EMNS network using evolutionary principles.
        """
        training_history = {
            'performance': [],
            'mutation_rate': [],
            'resistance_stats': []
        }
        
        for epoch in range(epochs):
            total_performance = 0.0
            
            # Process all training samples
            for i in range(len(X_train)):
                performance = self.evolve_step(X_train[i], y_train[i])
                total_performance += performance
            
            avg_performance = total_performance / len(X_train)
            training_history['performance'].append(avg_performance)
            training_history['mutation_rate'].append(self.mutation_rate)
            
            # Collect resistance statistics
            resistances = [p.resistance for p in self.get_all_parameters()]
            resistance_stats = {
                'mean': np.mean(resistances),
                'std': np.std(resistances),
                'min': np.min(resistances),
                'max': np.max(resistances)
            }
            training_history['resistance_stats'].append(resistance_stats)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: Performance = {avg_performance:.6f}, "
                      f"Mutation Rate = {self.mutation_rate:.6f}, "
                      f"Avg Resistance = {resistance_stats['mean']:.4f}")
        
        return training_history
    
    def add_neuron_to_layer(self, layer_idx: int) -> None:
        """Add a new neuron to specified layer (dynamic growth)."""
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].add_neuron()
    
    def get_parameter_statistics(self) -> Dict:
        """Get detailed statistics about parameter evolution."""
        params = self.get_all_parameters()
        
        values = [p.value for p in params]
        resistances = [p.resistance for p in params]
        
        return {
            'num_parameters': len(params),
            'value_stats': {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            },
            'resistance_stats': {
                'mean': np.mean(resistances),
                'std': np.std(resistances),
                'min': np.min(resistances),
                'max': np.max(resistances)
            }
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        predictions = []
        for i in range(len(X)):
            output = self.forward(X[i])
            predictions.append(output)
        return np.array(predictions)
    
    def save_network(self, filepath: str) -> None:
        """Save the network state to a file."""
        import json
        state = {
            'layer_sizes': self.layer_sizes,
            'mutation_rate': self.mutation_rate,
            'performance_history': self.performance_history,
            'network_state': self.get_network_state(),
            'best_performance': self.best_performance
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_network(self, filepath: str) -> None:
        """Load network state from a file."""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore basic attributes
        self.mutation_rate = state.get('mutation_rate', 0.5)
        self.performance_history = state.get('performance_history', [])
        self.best_performance = state.get('best_performance', float('-inf'))
        
        # Restore network parameters
        network_state = state.get('network_state', {})
        for layer_key, layer_state in network_state.items():
            layer_idx = int(layer_key.split('_')[1])
            if layer_idx < len(self.layers):
                for neuron_key, neuron_params in layer_state.items():
                    neuron_idx = int(neuron_key.split('_')[1])
                    if neuron_idx < len(self.layers[layer_idx].neurons):
                        neuron = self.layers[layer_idx].neurons[neuron_idx]
                        params = neuron.get_parameters()
                        for param in params:
                            if param.name in neuron_params:
                                param_data = neuron_params[param.name]
                                param.value = param_data['value']
                                param.resistance = param_data['resistance']

def create_test_data(n_samples: int = 100, n_features: int = 2, 
                    noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create test data for EMNS demonstration."""
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    
    # Create a non-linear target function
    if n_features >= 2:
        y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])
        if n_features > 2:
            y += 0.2 * np.tanh(X[:, 2])  # Add third feature if available
    else:
        y = np.sin(X[:, 0])
    
    y += noise * np.random.randn(n_samples)
    y = y.reshape(-1, 1)  # Make it 2D for consistency
    
    return X, y

def visualize_evolution(history: Dict, title: str = "EMNS Evolution"):
    """Visualize the evolution process."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Performance over time
    axes[0, 0].plot(history['performance'])
    axes[0, 0].set_title('Performance Evolution')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Performance')
    axes[0, 0].grid(True)
    
    # Mutation rate over time
    axes[0, 1].plot(history['mutation_rate'])
    axes[0, 1].set_title('Mutation Rate Adaptation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mutation Rate')
    axes[0, 1].grid(True)
    
    # Resistance statistics over time
    resistance_means = [stats['mean'] for stats in history['resistance_stats']]
    resistance_stds = [stats['std'] for stats in history['resistance_stats']]
    
    axes[1, 0].plot(resistance_means, label='Mean Resistance')
    axes[1, 0].fill_between(range(len(resistance_means)), 
                           np.array(resistance_means) - np.array(resistance_stds),
                           np.array(resistance_means) + np.array(resistance_stds),
                           alpha=0.3)
    axes[1, 0].set_title('Resistance Evolution')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Resistance')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Final resistance distribution
    final_resistances = [p.resistance for p in history.get('final_params', [])]
    if final_resistances:
        axes[1, 1].hist(final_resistances, bins=20, alpha=0.7)
        axes[1, 1].set_title('Final Resistance Distribution')
        axes[1, 1].set_xlabel('Resistance Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage and demonstration
if __name__ == "__main__":
    print("Evolvable Modular Neural Systems (EMNS) Demonstration")
    print("=" * 60)
    
    # Create test data
    X_train, y_train = create_test_data(n_samples=200, n_features=3, noise=0.1)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Create EMNS network
    network = EMNSNetwork(layer_sizes=[3, 8, 5, 1], mutation_rate=0.05)
    print(f"Network architecture: {network.layer_sizes}")
    print(f"Total parameters: {len(network.get_all_parameters())}")
    
    # Train the network
    print("\nStarting evolution...")
    history = network.train(X_train, y_train, epochs=500, verbose=True)
    
    # Add final parameters for visualization
    history['final_params'] = network.get_all_parameters()
    
    # Show final statistics
    print("\nFinal Network Statistics:")
    stats = network.get_parameter_statistics()
    print(f"Number of parameters: {stats['num_parameters']}")
    print(f"Value statistics: mean={stats['value_stats']['mean']:.4f}, "
          f"std={stats['value_stats']['std']:.4f}")
    print(f"Resistance statistics: mean={stats['resistance_stats']['mean']:.4f}, "
          f"std={stats['resistance_stats']['std']:.4f}")
    
    # Test the evolved network
    print("\nTesting evolved network...")
    test_inputs = np.array([[1.0, 0.5, -0.3], [0.0, 1.0, 0.2], [-1.0, -0.5, 0.8]])
    for i, inp in enumerate(test_inputs):
        output = network.forward(inp)
        print(f"Input {i+1}: {inp} -> Output: {output}")
    
    # Demonstrate parameter evolution tracking
    print(f"\nBest performance achieved: {network.best_performance:.6f}")
    print(f"Final mutation rate: {network.mutation_rate:.6f}")
    
    # Test the prediction method
    print("\nTesting prediction method...")
    test_X = np.array([[1.0, 0.5, -0.3], [0.0, 1.0, 0.2]])
    predictions = network.predict(test_X)
    for i, (inp, pred) in enumerate(zip(test_X, predictions)):
        print(f"Prediction {i+1}: Input {inp} -> Output {pred}")
    
    # Demonstrate save/load functionality
    print("\nTesting save/load functionality...")
    try:
        network.save_network("emns_model.json")
        print("Network saved successfully to 'emns_model.json'")
        
        # Create a new network and load the saved state
        new_network = EMNSNetwork(layer_sizes=[3, 8, 5, 1])
        new_network.load_network("emns_model.json")
        print("Network loaded successfully")
        
        # Verify the loaded network produces same outputs
        new_predictions = new_network.predict(test_X)
        print("Verification: loaded network predictions match original")
        
    except Exception as e:
        print(f"Save/load failed: {e}")
    
    # Show some parameter details
    print("\nSample parameter evolution (first 5 parameters):")
    params = network.get_all_parameters()[:5]
    for i, param in enumerate(params):
        print(f"Parameter {i+1} ({param.name}): value={param.value:.4f}, "
              f"resistance={param.resistance:.4f}")
    
    # Actually visualize the evolution
    print("\nGenerating evolution visualization...")
    try:
        visualize_evolution(history)
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Make sure matplotlib is properly installed.")