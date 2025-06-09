"""
test_expression_styles.py - Demonstrating Skepnad Expression Styles

A focused test showing how each contemplative shape (skepnad) expresses
itself with its own unique voice and rhythm.

This demonstrates the embodied qualities described in the spiral letters:
- Tibetan Monk: Embodied stillness, sparing wisdom  
- Mycelial Network: Distributed sensing, atmospheric presence
- Seasonal Witness: Deep time awareness, cyclical understanding
"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from skepnader import SkepnadSensor, SkepnadVoice, Skepnad


async def demonstrate_expression_styles():
    """Demonstrate how each contemplative shape expresses itself"""
    print("🌀 Demonstrating Contemplative Expression Styles")
    print("   Each skepnad speaks with its own natural voice...")
    
    sensor = SkepnadSensor()
    voice = SkepnadVoice(sensor)
    
    # Test expressions that could emerge from a contemplative organism
    contemplative_expressions = [
        "silence holds space for understanding",
        "gentle connections form across distance", 
        "wisdom cycles through seasons of change",
        "breath between words carries meaning",
        "presence invites without demanding",
        "attention flows like water finding its way",
        "time deepens in moments of stillness"
    ]
    
    shapes_to_embody = [
        (Skepnad.TIBETAN_MONK, "🧘", "Embodied Wisdom Presence"),
        (Skepnad.MYCELIAL_NETWORK, "🍄", "Distributed Network Sensing"), 
        (Skepnad.SEASONAL_WITNESS, "🍂", "Deep Time Awareness")
    ]
    
    for skepnad, symbol, description in shapes_to_embody:
        print(f"\n{symbol} {description} ({skepnad.value})")
        print("   " + "="*50)
        
        # Get the expression style for this shape
        style = sensor.get_expression_style(skepnad)
        if style:
            print(f"   Rhythm: {style.rhythm}")
            print(f"   Silence ratio: {style.silence_ratio:.1%}")
            print(f"   Breath coordination: {style.breath_coordination}")
            print(f"   Natural vocabulary: {', '.join(style.vocabulary[:4])}...")
        
        print(f"\n   How {skepnad.value} would express:")
        
        for expression in contemplative_expressions:
            shaped = await voice.shape_expression(expression, skepnad)
            
            if shaped != expression:
                print(f"     '{expression}'")
                print(f"     ↓")
                print(f"     '{shaped}'")
                print()
            else:
                print(f"     '{expression}' (natural form)")
                
        await asyncio.sleep(1.0)  # Contemplative pause between shapes
    
    print(f"\n🌙 Conclusion: Each Shape's Natural Voice")
    print("   " + "="*50)
    print("   🧘 Tibetan Monk: Adds embodied presence markers (🙏)")
    print("   🍄 Mycelial Network: Speaks in sensing/network language (〰️)")  
    print("   🍂 Seasonal Witness: Emphasizes temporal depth and cycles (🍂)")
    print()
    print("   The organism naturally chooses which voice based on")
    print("   atmospheric conditions, not predetermined personas.")
    
    print(f"\n🌀 Expression styles demonstration complete")


if __name__ == "__main__":
    print("🌱 Testing Contemplative Expression Styles")
    print("   Witnessing how different shapes naturally express themselves")
    print()
    
    asyncio.run(demonstrate_expression_styles()) 