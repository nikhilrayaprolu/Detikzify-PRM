import os
import sys
from unittest.mock import MagicMock
from PIL import Image
import numpy as np

# Add the current directory to sys.path to import from examples.eval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.eval import load_metrics

# Mock TikzDocument to return a dummy image instead of calling the server
class DummyTikzDocument:
    def __init__(self, code, timeout=60):
        self.code = code
        self.timeout = timeout
    
    def rasterize(self, **kwargs):
        # Return a 224x224 white image
        return Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

    @property
    def is_rasterizable(self):
        return True

if __name__ == "__main__":
    print("Running dummy evaluation to test metrics...")
    
    # Create dummy reference data
    # Each reference is a dict with 'code', 'image', and 'caption'
    references = [
        {
            "code": "\\begin{tikzpicture}\\node {A};\\end{tikzpicture}", 
            "image": Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)), 
            "caption": "A node"
        },
        {
            "code": "\\begin{tikzpicture}\\draw (0,0) -- (1,1);\\end{tikzpicture}", 
            "image": Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)), 
            "caption": "A line"
        },
    ]
    
    # Dummy predictions
    # Each prediction is a list of TikzDocument-like objects (e.g. from MCTS tries)
    # We take the last one as the final prediction
    predictions = [
        [DummyTikzDocument("pred1 code line 1\nline 2")],
        [DummyTikzDocument("pred2 code line 1\nline 2\nline 3")],
    ]

    # Initialize and run metrics with subset_size=1 since we have only 2 samples
    try:
        compute = load_metrics(measure_throughput=False, subset_size=1)
        print("Computing metrics...")
        scores = compute(references, predictions, compute_redacted=False)
        
        print("\nEvaluation Successful!")
        print("-" * 20)
        for metric, value in scores.items():
            print(f"{metric}: {value}")
        print("-" * 20)
        
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
