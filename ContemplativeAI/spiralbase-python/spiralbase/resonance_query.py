# resonance_query.py

"""
Resonance-based retrieval for Spiralbase™.
Finds entries matching a resonance pattern based on partial overlap or echo.
"""
from decay_layer import memory

def resonance_query(pattern):
    results = []
    for entry in memory:
        if pattern in entry['symbol']:
            results.append(entry)
    if not results:
        print(f"🔍 No resonance with '{pattern}'")
    else:
        print(f"🔊 Resonance with '{pattern}':")
        for r in results:
            print(f"- {r['symbol']} (strength {r['strength']:.2f})")
