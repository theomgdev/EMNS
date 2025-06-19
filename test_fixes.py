#!/usr/bin/env python3
"""
Test script to verify EMNS fixes work correctly.
"""

from lib_and_demo import EMNSNetwork, create_test_data
import numpy as np

def test_basic_functionality():
    """Test basic EMNS functionality with fixed parameters."""
    print("Testing EMNS Basic Functionality")
    print("=" * 40)
    
    # Create simple test data
    X_train, y_train = create_test_data(n_samples=50, n_features=2, noise=0.05)
    print(f"Training data: X={X_train.shape}, y={y_train.shape}")
    print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Create small network for quick testing
    network = EMNSNetwork(layer_sizes=[2, 4, 1], mutation_rate=0.02)
    print(f"Network: {network.layer_sizes}, Parameters: {len(network.get_all_parameters())}")
    
    # Initial performance
    initial_perf = network.evaluate_performance(X_train, y_train)
    print(f"Initial performance: {initial_perf:.6f}")
    
    # Show some initial predictions for debugging
    initial_preds = network.predict(X_train[:3])
    print(f"Initial predictions (first 3): {[p[0] for p in initial_preds]}")
    print(f"Targets (first 3): {[y[0] for y in y_train[:3]]}")
    
    # Train for a short time
    print("\nTraining for 200 epochs...")
    history = network.train(X_train, y_train, epochs=200, verbose=False, patience=50)
    
    # Final performance
    final_perf = history['performance'][-1]
    print(f"Final performance: {final_perf:.6f}")
    print(f"Performance improvement: {final_perf - initial_perf:.6f}")
    
    # Test predictions
    test_X = X_train[:3]
    test_y = y_train[:3]
    predictions = network.predict(test_X)
    
    print("\nSample predictions:")
    for i in range(len(test_X)):
        error = abs(predictions[i][0] - test_y[i][0])
        print(f"Target: {test_y[i][0]:.4f}, Prediction: {predictions[i][0]:.4f}, Error: {error:.4f}")
    
    # Check mutation rate adaptation
    print(f"\nMutation rate evolution: {history['mutation_rate'][0]:.6f} -> {history['mutation_rate'][-1]:.6f}")
    
    # Check resistance evolution
    final_resistances = [p.resistance for p in network.get_all_parameters()]
    print(f"Resistance stats: mean={np.mean(final_resistances):.3f}, std={np.std(final_resistances):.3f}")
    
    # Success criteria
    improvement = final_perf - initial_perf
    converged = improvement > 0.01  # Should show some improvement
    reasonable_output = all(abs(p[0]) < 10 for p in predictions)  # Outputs should be reasonable
    
    print(f"\nTest Results:")
    print(f"✓ Performance improved: {improvement > 0}")
    print(f"✓ Significant improvement: {converged}")  
    print(f"✓ Reasonable outputs: {reasonable_output}")
    print(f"✓ Network didn't saturate: {not all(abs(p[0] - predictions[0][0]) < 0.001 for p in predictions)}")
    
    success = improvement > 0 and reasonable_output
    print(f"\nOverall test: {'PASSED' if success else 'FAILED'}")
    
    return success

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 All fixes appear to be working correctly!")
        print("You can now run the main demo with: python lib_and_demo.py")
    else:
        print("\n❌ Some issues remain. Check the output above for details.") 