import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from vector_store import VectorStore

def test_filtering():
    print("Testing Vector Store Filtering...")
    v = VectorStore()
    
    docs = [
        {'text': 'Apple is a fruit.', 'source': 'fruit.txt'},
        {'text': 'Tesla makes electric cars.', 'source': 'cars.txt'},
        {'text': 'Banana is yellow.', 'source': 'fruit.txt'}
    ]
    
    v.build_index(docs)
    
    print("\n1. Global search for 'fruit':")
    results = v.search("fruit", top_k=2)
    for r in results:
        print(f" - {r['source']}: {r['text']}")
    
    print("\n2. Targeted search in 'cars.txt' for 'fruit':")
    results = v.search("fruit", top_k=2, source='cars.txt')
    for r in results:
        print(f" - {r['source']}: {r['text']}")
        assert r['source'] == 'cars.txt'
    
    print("\n3. Targeted search in 'fruit.txt' for 'Tesla':")
    results = v.search("Tesla", top_k=2, source='fruit.txt')
    for r in results:
        print(f" - {r['source']}: {r['text']}")
        assert r['source'] == 'fruit.txt'

    print("\n✓ Verification successful!")

if __name__ == "__main__":
    test_filtering()
