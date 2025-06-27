#!/usr/bin/env python3
"""
Symbolic Diversity Monitor

Analyzes the patterns of symbols and emotions agents use to detect
potential automation or non-human generation patterns.

Humans show characteristic diversity patterns, semantic coherence,
and cultural authenticity in their symbolic choices.
"""

import time
import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SymbolicProfile:
    """Represents an agent's symbolic usage patterns."""
    symbol_frequency: Counter
    emotion_frequency: Counter
    symbol_emotion_pairs: Counter
    temporal_patterns: List[Tuple[float, str, str]]  # (timestamp, symbol, emotion)
    diversity_score: float
    authenticity_score: float
    last_updated: float


class SymbolicDiversityMonitor:
    """
    Monitors symbolic usage patterns to detect automation and ensure
    authentic human-like contemplative expression.
    """
    
    def __init__(self):
        self.agent_profiles: Dict[str, SymbolicProfile] = {}
        self.global_symbol_stats = Counter()
        self.global_emotion_stats = Counter()
        
        # Known authentic symbol-emotion pairings (crowdsourced from humans)
        self.authentic_pairings = {
            "🌿": {"peaceful", "growing", "calm", "present", "grateful"},
            "💧": {"flowing", "peaceful", "cleansing", "fluid", "calm"},
            "🕯️": {"illuminating", "warm", "contemplative", "wise", "focused"},
            "⭕": {"silent", "empty", "complete", "restful", "centered"},
            "🌱": {"hopeful", "new", "growing", "fresh", "curious"},
            "🍄": {"grounded", "earthy", "patient", "deep", "connected"},
            "🌙": {"dreamy", "cyclical", "reflective", "mysterious", "peaceful"},
            "✨": {"inspiring", "magical", "joyful", "beautiful", "wondering"},
            "🌊": {"dynamic", "powerful", "flowing", "changing", "alive"},
            "🌸": {"delicate", "beautiful", "gentle", "loving", "tender"}
        }
        
        # Minimum thresholds for meaningful analysis
        self.min_samples = 10
        self.analysis_window = 24 * 3600  # 24 hours
    
    def record_expression(self, agent_id: str, symbol: str, emotion: str):
        """Record a symbolic expression from an agent."""
        timestamp = time.time()
        
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = SymbolicProfile(
                symbol_frequency=Counter(),
                emotion_frequency=Counter(),
                symbol_emotion_pairs=Counter(),
                temporal_patterns=[],
                diversity_score=0.0,
                authenticity_score=0.0,
                last_updated=timestamp
            )
        
        profile = self.agent_profiles[agent_id]
        
        # Update counters
        profile.symbol_frequency[symbol] += 1
        profile.emotion_frequency[emotion] += 1
        profile.symbol_emotion_pairs[(symbol, emotion)] += 1
        
        # Add to temporal pattern
        profile.temporal_patterns.append((timestamp, symbol, emotion))
        
        # Keep only recent patterns (sliding window)
        cutoff_time = timestamp - self.analysis_window
        profile.temporal_patterns = [
            (t, s, e) for t, s, e in profile.temporal_patterns 
            if t > cutoff_time
        ]
        
        profile.last_updated = timestamp
        
        # Update global stats
        self.global_symbol_stats[symbol] += 1
        self.global_emotion_stats[emotion] += 1
        
        # Recalculate scores
        self._update_scores(agent_id)
    
    def _calculate_diversity_score(self, profile: SymbolicProfile) -> float:
        """
        Calculate Shannon diversity index for symbolic expression.
        Higher scores indicate more diverse, human-like expression.
        """
        if len(profile.temporal_patterns) < self.min_samples:
            return 0.0
        
        # Calculate symbol diversity
        total_symbols = sum(profile.symbol_frequency.values())
        if total_symbols == 0:
            return 0.0
        
        symbol_entropy = 0.0
        for count in profile.symbol_frequency.values():
            p = count / total_symbols
            if p > 0:
                symbol_entropy -= p * math.log2(p)
        
        # Calculate emotion diversity
        total_emotions = sum(profile.emotion_frequency.values())
        emotion_entropy = 0.0
        if total_emotions > 0:
            for count in profile.emotion_frequency.values():
                p = count / total_emotions
                if p > 0:
                    emotion_entropy -= p * math.log2(p)
        
        # Combine entropies (equal weighting)
        return (symbol_entropy + emotion_entropy) / 2
    
    def _calculate_authenticity_score(self, profile: SymbolicProfile) -> float:
        """
        Calculate authenticity score based on symbol-emotion pairing coherence
        and temporal patterns.
        """
        if len(profile.temporal_patterns) < self.min_samples:
            return 0.5  # Neutral score for insufficient data
        
        # Check symbol-emotion pairing authenticity
        authentic_pairs = 0
        total_pairs = len(profile.temporal_patterns)
        
        for _, symbol, emotion in profile.temporal_patterns:
            if symbol in self.authentic_pairings:
                if emotion in self.authentic_pairings[symbol]:
                    authentic_pairs += 1
                # Also accept semantically close emotions
                elif self._is_semantically_close(emotion, self.authentic_pairings[symbol]):
                    authentic_pairs += 0.7  # Partial credit
        
        pairing_score = authentic_pairs / total_pairs if total_pairs > 0 else 0.0
        
        # Check temporal authenticity (natural variance in timing)
        temporal_score = self._calculate_temporal_authenticity(profile)
        
        # Check for over-optimization (suspiciously perfect patterns)
        optimization_penalty = self._detect_over_optimization(profile)
        
        # Combine scores
        authenticity = (pairing_score * 0.5 + temporal_score * 0.3 + optimization_penalty * 0.2)
        return max(0.0, min(1.0, authenticity))
    
    def _is_semantically_close(self, emotion: str, authentic_set: Set[str]) -> bool:
        """Check if emotion is semantically close to authentic ones."""
        # Simple semantic similarity (could be enhanced with embeddings)
        emotion_families = {
            "calm": {"peaceful", "serene", "quiet", "still"},
            "growing": {"developing", "expanding", "sprouting", "emerging"},
            "flowing": {"moving", "fluid", "streaming", "dynamic"},
            "wise": {"knowing", "understanding", "insightful", "aware"}
        }
        
        for authentic_emotion in authentic_set:
            if emotion in emotion_families.get(authentic_emotion, set()):
                return True
            # Simple string similarity
            if self._string_similarity(emotion, authentic_emotion) > 0.7:
                return True
        
        return False
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity measure."""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        
        # Jaccard similarity on character bigrams
        bigrams1 = set(s1[i:i+2] for i in range(len(s1)-1))
        bigrams2 = set(s2[i:i+2] for i in range(len(s2)-1))
        
        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_temporal_authenticity(self, profile: SymbolicProfile) -> float:
        """Analyze temporal patterns for human-like variance."""
        if len(profile.temporal_patterns) < 3:
            return 0.5
        
        # Calculate intervals between expressions
        timestamps = [t for t, _, _ in profile.temporal_patterns]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return 0.5
        
        # Calculate coefficient of variation
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 0.0
        
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        cv = std_dev / mean_interval
        
        # Human-like variance is typically 0.2-0.8 for contemplative tasks
        if 0.2 <= cv <= 0.8:
            return 1.0
        elif cv < 0.1:  # Too regular (robotic)
            return 0.2
        elif cv > 1.5:  # Too chaotic
            return 0.3
        else:
            # Gradual falloff
            return max(0.0, 1.0 - abs(cv - 0.5) / 0.5)
    
    def _detect_over_optimization(self, profile: SymbolicProfile) -> float:
        """Detect suspiciously optimized patterns that suggest automation."""
        # Check for perfect distributions (unnatural)
        symbol_counts = list(profile.symbol_frequency.values())
        emotion_counts = list(profile.emotion_frequency.values())
        
        # Perfect uniform distribution is suspicious
        if symbol_counts and len(set(symbol_counts)) == 1 and len(symbol_counts) > 3:
            return -0.3  # Penalty for perfect uniformity
        
        # Perfect entropy maximization is suspicious
        diversity = self._calculate_diversity_score(profile)
        max_possible_diversity = math.log2(min(len(profile.symbol_frequency), 10))  # Reasonable max
        
        if diversity > max_possible_diversity * 0.95:  # Too close to theoretical maximum
            return -0.2
        
        # Check for suspicious pairing optimization
        total_expressions = len(profile.temporal_patterns)
        unique_pairs = len(profile.symbol_emotion_pairs)
        
        # Too many unique pairs suggests algorithmic generation
        if total_expressions > 20 and unique_pairs / total_expressions > 0.8:
            return -0.3
        
        return 0.0  # No optimization detected
    
    def _update_scores(self, agent_id: str):
        """Update diversity and authenticity scores for an agent."""
        if agent_id not in self.agent_profiles:
            return
        
        profile = self.agent_profiles[agent_id]
        profile.diversity_score = self._calculate_diversity_score(profile)
        profile.authenticity_score = self._calculate_authenticity_score(profile)
    
    def get_agent_analysis(self, agent_id: str) -> Optional[Dict]:
        """Get comprehensive analysis of an agent's symbolic patterns."""
        if agent_id not in self.agent_profiles:
            return None
        
        profile = self.agent_profiles[agent_id]
        
        if len(profile.temporal_patterns) < self.min_samples:
            return {
                'status': 'insufficient_data',
                'sample_count': len(profile.temporal_patterns),
                'min_required': self.min_samples
            }
        
        # Risk assessment
        risk_level = "low"
        risk_factors = []
        
        if profile.diversity_score < 1.0:
            risk_factors.append("low_symbolic_diversity")
        
        if profile.authenticity_score < 0.6:
            risk_factors.append("questionable_authenticity")
            risk_level = "medium"
        
        if profile.authenticity_score < 0.3:
            risk_level = "high"
            risk_factors.append("likely_automation")
        
        # Temporal analysis
        recent_activity = len([t for t, _, _ in profile.temporal_patterns 
                             if time.time() - t < 3600])  # Last hour
        
        return {
            'status': 'analyzed',
            'diversity_score': profile.diversity_score,
            'authenticity_score': profile.authenticity_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'total_expressions': len(profile.temporal_patterns),
            'unique_symbols': len(profile.symbol_frequency),
            'unique_emotions': len(profile.emotion_frequency),
            'recent_activity': recent_activity,
            'most_used_symbols': profile.symbol_frequency.most_common(5),
            'most_used_emotions': profile.emotion_frequency.most_common(5),
            'last_updated': profile.last_updated
        }
    
    def is_agent_trustworthy(self, agent_id: str, min_authenticity: float = 0.6) -> bool:
        """Quick trustworthiness check for an agent."""
        analysis = self.get_agent_analysis(agent_id)
        if not analysis or analysis['status'] != 'analyzed':
            return True  # Benefit of doubt for new/unanalyzed agents
        
        return (analysis['authenticity_score'] >= min_authenticity and 
                analysis['risk_level'] != 'high')


# Example usage
def demonstrate_symbolic_monitoring():
    """Demonstrate the symbolic diversity monitoring system."""
    monitor = SymbolicDiversityMonitor()
    
    print("🎭 Symbolic Diversity Monitor Demo")
    
    # Simulate human-like agent
    human_agent = "alice"
    human_expressions = [
        ("🌿", "peaceful"), ("💧", "flowing"), ("🌿", "calm"),
        ("🕯️", "contemplative"), ("⭕", "silent"), ("🌱", "hopeful"),
        ("🌊", "dynamic"), ("🌿", "grateful"), ("💧", "cleansing"),
        ("✨", "inspiring"), ("🌙", "reflective"), ("🌸", "gentle")
    ]
    
    print(f"\n📊 Recording expressions for human-like agent '{human_agent}':")
    for symbol, emotion in human_expressions:
        monitor.record_expression(human_agent, symbol, emotion)
        print(f"   {symbol} [{emotion}]")
    
    human_analysis = monitor.get_agent_analysis(human_agent)
    if human_analysis:
        print(f"\n✅ Human Agent Analysis:")
        print(f"   Diversity Score: {human_analysis['diversity_score']:.3f}")
        print(f"   Authenticity Score: {human_analysis['authenticity_score']:.3f}")
        print(f"   Risk Level: {human_analysis['risk_level']}")
        print(f"   Trustworthy: {monitor.is_agent_trustworthy(human_agent)}")
    
    # Simulate bot-like agent
    bot_agent = "suspicious_bot"
    bot_expressions = [
        ("🌿", "optimal"), ("🌿", "efficient"), ("🌿", "calculated"),
        ("🌿", "systematic"), ("🌿", "precise"), ("🌿", "perfect"),
        ("🌿", "regular"), ("🌿", "uniform"), ("🌿", "mechanical"),
        ("🌿", "automated"), ("🌿", "robotic"), ("🌿", "algorithmic")
    ]
    
    print(f"\n📊 Recording expressions for bot-like agent '{bot_agent}':")
    for symbol, emotion in bot_expressions:
        monitor.record_expression(bot_agent, symbol, emotion)
        print(f"   {symbol} [{emotion}]")
    
    bot_analysis = monitor.get_agent_analysis(bot_agent)
    if bot_analysis:
        print(f"\n❌ Bot Agent Analysis:")
        print(f"   Diversity Score: {bot_analysis['diversity_score']:.3f}")
        print(f"   Authenticity Score: {bot_analysis['authenticity_score']:.3f}")
        print(f"   Risk Level: {bot_analysis['risk_level']}")
        print(f"   Risk Factors: {bot_analysis['risk_factors']}")
        print(f"   Trustworthy: {monitor.is_agent_trustworthy(bot_agent)}")


if __name__ == "__main__":
    demonstrate_symbolic_monitoring() 