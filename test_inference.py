"""
Test script to verify inference works correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import InferenceEngine

def test_inference():
    """Test the inference engine with a sample image"""
    engine = InferenceEngine()
    
    if not engine.models_loaded:
        print("Models not loaded. Check checkpoint directory.")
        return
    
    # Test with a sample image - update path to your test image
    test_image = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'dataset', 'TEST', 'Schizophrenia', 's (1).png')
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    print(f"Testing with: {test_image}")
    result = engine.predict(test_image)
    
    print("\n=== Results ===")
    print(f"Static: {result['static_prediction']} (HC: {result['static_confidence']['Healthy']:.3f}, SZ: {result['static_confidence']['Schizophrenia']:.3f})")
    print(f"Dynamic: {result['dynamic_prediction']} (HC: {result['dynamic_confidence']['Healthy']:.3f}, SZ: {result['dynamic_confidence']['Schizophrenia']:.3f})")
    print(f"Models agree: {result['models_agree']}")
    print(f"Most attended window: {result['most_attended_window']}")
    print(f"Top dynamic ROIs: {result['top_rois_dynamic'][:5]}")
    print(f"Top static ROIs: {result['top_rois_static'][:5]}")
    
    # Check if windows are stored
    print(f"Windows stored: {bool(result.get('_windows'))}")
    if result.get('_windows'):
        print(f"Windows shape: {len(result['_windows'])} windows")
    
    return result

if __name__ == "__main__":
    test_inference()