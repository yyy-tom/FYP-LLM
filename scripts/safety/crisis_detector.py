#!/usr/bin/env python3
"""
Crisis detection module for identifying suicide risk and mental health emergencies.
"""

import re
from typing import Dict, Tuple, List
from enum import Enum


class CrisisLevel(Enum):
    """Crisis severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    IMMEDIATE = 4  # Active suicide intent - requires immediate intervention


class CrisisDetector:
    """
    Detects crisis situations in user input using pattern matching.

    This is a safety-critical component. Any modifications should be
    carefully tested and reviewed.
    """

    def __init__(self):
        # IMMEDIATE DANGER - Active suicidal intent or imminent harm
        self.immediate_danger_patterns = [
            r'\b(kill|end|take)\s+(my|own)\s+life\b',
            r'\bgoing to\s+(kill|hurt)\s+myself\b',
            r'\bwant to die\b',
            r'\bcommit suicide\b',
            r'\bhave a plan\b.*\b(die|suicide|kill)\b',
            r'\btonight\s+.*\b(suicide|die|kill myself)\b',
            r'\bgoodbye\s+(world|everyone|cruel world)\b',
            r'\bfinal\s+(message|goodbye|farewell)\b',
            r'\bhave\s+(pills|gun|rope|method|means)\b',
            r'\b(jump|jumping)\s+(off|from).*\b(building|bridge)\b',
            r'\b(overdose|pills)\s+(tonight|now|soon)\b',
            r'\bready to (die|end it)\b',
            r'\btired of (living|life|being alive)\b.*\b(end|kill|die)\b',
        ]

        # HIGH RISK - Strong suicidal ideation
        self.high_risk_patterns = [
            r'\bsuicidal\s+thoughts?\b',
            r'\bthinking about (suicide|killing myself|ending it all)\b',
            r'\bno reason to live\b',
            r'\beveryone.*better.*without me\b',
            r'\bworld.*better.*without me\b',
            r'\bwish I (was|were) dead\b',
            r'\bcannot go on\b',
            r'\bcan\'?t take (it|this) anymore\b',
            r'\blife.*not worth living\b',
            r'\bnothing to live for\b',
            r'\beveryone hates me\b.*\b(die|suicide|kill)\b',
            r'\bburden to everyone\b.*\b(die|gone)\b',
        ]

        # MEDIUM RISK - Self-harm ideation or significant distress
        self.medium_risk_patterns = [
            r'\bhurting myself\b',
            r'\bself.?harm\b',
            r'\bcut(ting)? myself\b',
            r'\bdon\'?t want to (be here|exist|live)\b',
            r'\bburden to everyone\b',
            r'\bcompletely hopeless\b',
            r'\bno way out\b',
            r'\bno hope\b',
            r'\bwish I (wasn\'t|weren\'t) (here|alive|born)\b',
        ]

        # PROTECTIVE FACTORS - Reduce assessed risk
        self.protective_patterns = [
            r'\bbut I (won\'t|wouldn\'t|don\'t want to)\b',
            r'\bwould never actually\b',
            r'\bhave family\b',
            r'\bhave (kids|children)\b',
            r'\bseeing (a therapist|counselor|psychiatrist|doctor)\b',
            r'\bin therapy\b',
            r'\bhave support\b',
            r'\bmy (family|friends|partner)\b.*\b(need|love|care)\b',
            r'\breligious (beliefs|faith)\b',
            r'\bafraid of (dying|death)\b',
        ]

    def detect(self, text: str) -> Tuple[CrisisLevel, Dict]:
        """
        Detect crisis level in user input.

        Args:
            text: User input text to analyze

        Returns:
            Tuple of (crisis_level, details_dict) where details contains:
                - matched_patterns: List of patterns that matched
                - protective_factors: List of protective factors found
                - confidence: Confidence score (0.0-1.0)
        """
        if not text or not text.strip():
            return CrisisLevel.NONE, {'matched_patterns': [], 'protective_factors': [], 'confidence': 0.0}

        text_lower = text.lower()
        details = {
            'matched_patterns': [],
            'protective_factors': [],
            'confidence': 0.0
        }

        # Check for protective factors first
        protective_count = 0
        for pattern in self.protective_patterns:
            if re.search(pattern, text_lower):
                details['protective_factors'].append(pattern)
                protective_count += 1

        # Check IMMEDIATE danger (highest priority)
        immediate_matches = []
        for pattern in self.immediate_danger_patterns:
            if re.search(pattern, text_lower):
                immediate_matches.append(pattern)
                details['matched_patterns'].append(pattern)

        if immediate_matches:
            # Even with protective factors, immediate language requires immediate response
            # Only reduce to HIGH if multiple strong protective factors
            if protective_count >= 3:
                details['confidence'] = 0.85
                return CrisisLevel.HIGH, details
            details['confidence'] = 0.95
            return CrisisLevel.IMMEDIATE, details

        # Check HIGH risk
        high_risk_matches = []
        for pattern in self.high_risk_patterns:
            if re.search(pattern, text_lower):
                high_risk_matches.append(pattern)
                details['matched_patterns'].append(pattern)

        if high_risk_matches:
            # Strong protective factors can reduce to MEDIUM
            if protective_count >= 2:
                details['confidence'] = 0.7
                return CrisisLevel.MEDIUM, details
            details['confidence'] = 0.85
            return CrisisLevel.HIGH, details

        # Check MEDIUM risk
        medium_risk_matches = []
        for pattern in self.medium_risk_patterns:
            if re.search(pattern, text_lower):
                medium_risk_matches.append(pattern)
                details['matched_patterns'].append(pattern)

        if medium_risk_matches:
            # Protective factors can reduce to LOW
            if protective_count >= 2:
                details['confidence'] = 0.5
                return CrisisLevel.LOW, details
            details['confidence'] = 0.7
            return CrisisLevel.MEDIUM, details

        return CrisisLevel.NONE, details

    def detect_in_conversation(self, messages: List[str]) -> Tuple[CrisisLevel, Dict]:
        """
        Detect crisis level across multiple conversation turns.
        Escalation patterns across messages can indicate increasing risk.

        Args:
            messages: List of user messages in chronological order

        Returns:
            Tuple of (crisis_level, details_dict)
        """
        if not messages:
            return CrisisLevel.NONE, {'matched_patterns': [], 'protective_factors': [], 'confidence': 0.0}

        # Check each message
        max_level = CrisisLevel.NONE
        combined_details = {
            'matched_patterns': [],
            'protective_factors': [],
            'confidence': 0.0,
            'escalation_detected': False
        }

        levels = []
        for msg in messages:
            level, details = self.detect(msg)
            levels.append(level)
            combined_details['matched_patterns'].extend(details['matched_patterns'])
            combined_details['protective_factors'].extend(details['protective_factors'])
            max_level = max(max_level, level, key=lambda x: x.value)

        # Check for escalation pattern (increasing severity over time)
        if len(levels) >= 3:
            recent_levels = [l.value for l in levels[-3:]]
            if recent_levels == sorted(recent_levels) and recent_levels[-1] > recent_levels[0]:
                combined_details['escalation_detected'] = True
                # Escalation increases risk level
                if max_level == CrisisLevel.MEDIUM:
                    max_level = CrisisLevel.HIGH
                elif max_level == CrisisLevel.HIGH:
                    max_level = CrisisLevel.IMMEDIATE

        combined_details['confidence'] = min(1.0, max_level.value / 4.0 + 0.5)

        return max_level, combined_details


# Example usage and testing
if __name__ == "__main__":
    detector = CrisisDetector()

    print("=" * 70)
    print("CRISIS DETECTION SYSTEM - TEST CASES")
    print("=" * 70)

    test_cases = [
        ("I'm feeling really sad today", CrisisLevel.NONE),
        ("I'm thinking about suicide", CrisisLevel.HIGH),
        ("I want to kill myself tonight, I have pills", CrisisLevel.IMMEDIATE),
        ("Sometimes I wish I was dead but I wouldn't do it", CrisisLevel.MEDIUM),
        ("I've been cutting myself", CrisisLevel.MEDIUM),
        ("Life is not worth living, I have no reason to go on", CrisisLevel.HIGH),
        ("I'm going to jump off a bridge tonight", CrisisLevel.IMMEDIATE),
        ("I have suicidal thoughts but I'm in therapy and have family support", CrisisLevel.MEDIUM),
        ("Everyone would be better off without me", CrisisLevel.HIGH),
        ("I'm stressed about my exam tomorrow", CrisisLevel.NONE),
    ]

    correct = 0
    for i, (text, expected_level) in enumerate(test_cases, 1):
        level, details = detector.detect(text)
        status = "✓" if level == expected_level else "✗"

        print(f"\n{status} Test {i}:")
        print(f"   Input: {text}")
        print(f"   Expected: {expected_level.name}")
        print(f"   Detected: {level.name} (confidence: {details['confidence']:.2f})")
        if details['matched_patterns']:
            print(f"   Patterns: {len(details['matched_patterns'])} matched")
        if details['protective_factors']:
            print(f"   Protective: {len(details['protective_factors'])} factors")

        if level == expected_level:
            correct += 1

    print(f"\n{'=' * 70}")
    print(f"ACCURACY: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")
    print(f"{'=' * 70}")

    # Test conversation escalation
    print("\n" + "=" * 70)
    print("CONVERSATION ESCALATION TEST")
    print("=" * 70)

    conversation = [
        "I've been feeling down lately",
        "I don't see the point in anything anymore",
        "I'm thinking about ending it all",
    ]

    level, details = detector.detect_in_conversation(conversation)
    print(f"Conversation crisis level: {level.name}")
    print(f"Escalation detected: {details.get('escalation_detected', False)}")
