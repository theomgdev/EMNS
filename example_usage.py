#!/usr/bin/env python3
"""
Example usage of the EMNS (Evolvable Modular Neural Systems) library.

This example demonstrates basic usage, training, and evaluation of an EMNS network.
"""

import numpy as np
from lib_and_demo import EMNSNetwork, create_test_data, visualize_evolution

def simple_regression_example():
    """Demonstrate EMNS on a simple regression task."""
    print("EMNS Simple Regression Example")
    print("=" * 40)
    
    # Create synthetic data
    X_train, y_train = create_test_data(n_samples=100, n_features=2, noise=0.1)
    X_test, y_test = create_test_data(n_samples=20, n_features=2, noise=0.1)
    
    print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
    
    # Create and train network
    network = EMNSNetwork(layer_sizes=[2, 6, 4, 1], mutation_rate=0.01)
    print(f"Network created with {len(network.get_all_parameters())} parameters")
    
    # Train
    history = network.train(X_train, y_train, epochs=500, verbose=True, patience=50)
    
    # Evaluate
    predictions = network.predict(X_test)
    mse = np.mean((predictions.flatten() - y_test.flatten()) ** 2)
    print(f"\nTest MSE: {mse:.6f}")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(5, len(X_test))):
        print(f"Input: {X_test[i]}, Target: {y_test[i][0]:.4f}, Prediction: {predictions[i][0]:.4f}")
    
    return network, history

def classification_example():
    """Demonstrate EMNS on a simple classification task."""
    print("\n\nEMNS Classification Example")
    print("=" * 40)
    
    # Create synthetic classification data
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    
    # Simple classification rule: positive if x1 + x2 > 0
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
    
    # Create and train network
    network = EMNSNetwork(layer_sizes=[2, 8, 4, 1], mutation_rate=0.02)
    print(f"Network created with {len(network.get_all_parameters())} parameters")
    
    # Train
    history = network.train(X_train, y_train, epochs=600, verbose=True, patience=80)
    
    # Evaluate
    predictions = network.predict(X_test)
    binary_predictions = (predictions > 0.5).astype(float)
    accuracy = np.mean(binary_predictions.flatten() == y_test.flatten())
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Show resistance statistics
    stats = network.get_parameter_statistics()
    print(f"Final resistance statistics: mean={stats['resistance_stats']['mean']:.4f}, "
          f"std={stats['resistance_stats']['std']:.4f}")
    
    return network, history

def continual_learning_example():
    """Demonstrate continual learning capabilities."""
    print("\n\nEMNS Continual Learning Example")
    print("=" * 40)
    
    # Create network
    network = EMNSNetwork(layer_sizes=[2, 6, 1], mutation_rate=0.015)
    
    # Task 1: Learn sin(x1)
    print("Learning Task 1: sin(x1)")
    X1 = np.random.uniform(-np.pi, np.pi, (100, 2))
    y1 = 0.5 * np.tanh(np.sin(X1[:, 0])).reshape(-1, 1)  # Apply normalization
    
    history1 = network.train(X1, y1, epochs=300, verbose=False, patience=50)
    print(f"Task 1 final performance: {history1['performance'][-1]:.6f}")
    
    # Task 2: Learn cos(x2) while retaining sin(x1)
    print("Learning Task 2: cos(x2)")
    X2 = np.random.uniform(-np.pi, np.pi, (100, 2))
    y2 = 0.5 * np.tanh(np.cos(X2[:, 1])).reshape(-1, 1)  # Apply normalization
    
    history2 = network.train(X2, y2, epochs=300, verbose=False, patience=50)
    print(f"Task 2 final performance: {history2['performance'][-1]:.6f}")
    
    # Test retention of Task 1
    test_X1 = np.random.uniform(-np.pi, np.pi, (20, 2))
    test_y1 = 0.5 * np.tanh(np.sin(test_X1[:, 0])).reshape(-1, 1)  # Apply normalization
    pred1 = network.predict(test_X1)
    mse1 = np.mean((pred1 - test_y1) ** 2)
    print(f"Task 1 retention MSE: {mse1:.6f}")
    
    # Test Task 2 performance
    test_X2 = np.random.uniform(-np.pi, np.pi, (20, 2))
    test_y2 = 0.5 * np.tanh(np.cos(test_X2[:, 1])).reshape(-1, 1)  # Apply normalization
    pred2 = network.predict(test_X2)
    mse2 = np.mean((pred2 - test_y2) ** 2)
    print(f"Task 2 performance MSE: {mse2:.6f}")
    
    return network

if __name__ == "__main__":
    # Run examples
    network1, history1 = simple_regression_example()
    network2, history2 = classification_example()
    network3 = continual_learning_example()
    
    # Save an example network
    print("\n\nSaving trained network...")
    network1.save_network("example_trained_network.json")
    print("Network saved to 'example_trained_network.json'")
    
    # Visualize evolution for the regression example
    print("\nGenerating visualization for regression example...")
    try:
        history1['final_params'] = network1.get_all_parameters()
        visualize_evolution(history1, "EMNS Regression Example Evolution")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("\nAll examples completed successfully!") 