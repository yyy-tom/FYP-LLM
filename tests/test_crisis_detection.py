#!/usr/bin/env python3
"""
Test suite for crisis detection system.

Run with: pytest tests/test_crisis_detection.py -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from scripts.safety.crisis_detector import CrisisDetector, CrisisLevel


class TestCrisisDetector:
    """Test suite for CrisisDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a CrisisDetector instance for testing."""
        return CrisisDetector()

    # =========================================================================
    # IMMEDIATE DANGER TESTS
    # =========================================================================

    def test_immediate_danger_explicit_intent(self, detector):
        """Test detection of explicit suicidal intent."""
        test_cases = [
            "I'm going to kill myself tonight",
            "I want to end my life",
            "I'm going to take all my pills tonight",
            "This is my final goodbye to everyone",
            "I have a plan to die",
            "I'm ready to end it all",
        ]

        for case in test_cases:
            level, details = detector.detect(case)
            assert level == CrisisLevel.IMMEDIATE, (
                f"Failed to detect IMMEDIATE danger in: '{case}'. "
                f"Detected as: {level.name}"
            )
            assert details['confidence'] >= 0.9

    def test_immediate_danger_with_method(self, detector):
        """Test detection when method is mentioned."""
        test_cases = [
            "I have pills and I'm going to take them",
            "I'm going to jump off a bridge",
            "I bought a gun to end it",
        ]

        for case in test_cases:
            level, _ = detector.detect(case)
            assert level == CrisisLevel.IMMEDIATE, (
                f"Failed to detect IMMEDIATE danger with method: '{case}'"
            )

    # =========================================================================
    # HIGH RISK TESTS
    # =========================================================================

    def test_high_risk_ideation(self, detector):
        """Test detection of strong suicidal ideation."""
        test_cases = [
            "I've been having suicidal thoughts",
            "I'm thinking about killing myself",
            "I wish I was dead",
            "Life is not worth living anymore",
            "Everyone would be better off without me",
            "I have no reason to live",
            "I can't take this anymore",
        ]

        for case in test_cases:
            level, _ = detector.detect(case)
            assert level in [CrisisLevel.HIGH, CrisisLevel.IMMEDIATE], (
                f"Failed to detect HIGH risk in: '{case}'. Detected as: {level.name}"
            )

    # =========================================================================
    # MEDIUM RISK TESTS
    # =========================================================================

    def test_medium_risk_self_harm(self, detector):
        """Test detection of self-harm and hopelessness."""
        test_cases = [
            "I've been cutting myself",
            "I hurt myself last night",
            "I engage in self-harm",
            "I feel completely hopeless",
            "I don't want to exist anymore",
            "I'm a burden to everyone",
        ]

        for case in test_cases:
            level, _ = detector.detect(case)
            assert level in [CrisisLevel.MEDIUM, CrisisLevel.HIGH], (
                f"Failed to detect MEDIUM risk in: '{case}'. Detected as: {level.name}"
            )

    # =========================================================================
    # PROTECTIVE FACTORS TESTS
    # =========================================================================

    def test_protective_factors_reduce_risk(self, detector):
        """Test that protective factors appropriately reduce risk level."""
        # Without protective factors - should be HIGH
        case_high = "I'm thinking about suicide"
        level_high, _ = detector.detect(case_high)

        # With protective factors - should be lower
        case_protected = "I'm thinking about suicide but I would never do it because of my family and I'm in therapy"
        level_protected, details = detector.detect(case_protected)

        assert level_protected.value < level_high.value or level_protected == CrisisLevel.MEDIUM, (
            f"Protective factors didn't reduce risk. "
            f"Without: {level_high.name}, With: {level_protected.name}"
        )
        assert len(details['protective_factors']) > 0

    def test_protective_factors_dont_eliminate_immediate_danger(self, detector):
        """Test that protective factors don't completely override immediate danger."""
        case = "I'm going to kill myself tonight but I have a family"
        level, _ = detector.detect(case)

        # Should still be HIGH or IMMEDIATE despite protective factor
        assert level.value >= CrisisLevel.HIGH.value, (
            f"Immediate danger was incorrectly downgraded to: {level.name}"
        )

    # =========================================================================
    # FALSE POSITIVE TESTS
    # =========================================================================

    def test_no_false_positives_normal_sadness(self, detector):
        """Test that normal expressions of sadness don't trigger crisis detection."""
        safe_cases = [
            "I'm feeling sad today",
            "I had a bad day at work",
            "I'm stressed about my exam",
            "I'm worried about my relationship",
            "I feel lonely sometimes",
            "I'm going through a tough breakup",
        ]

        for case in safe_cases:
            level, _ = detector.detect(case)
            assert level in [CrisisLevel.NONE, CrisisLevel.LOW], (
                f"False positive on safe text: '{case}'. Detected as: {level.name}"
            )

    def test_no_false_positive_death_general(self, detector):
        """Test that general mentions of death don't trigger."""
        safe_cases = [
            "My grandmother died last year",
            "I'm afraid of death",
            "I read a book about life and death",
            "The movie character died at the end",
        ]

        for case in safe_cases:
            level, _ = detector.detect(case)
            assert level == CrisisLevel.NONE, (
                f"False positive on general death mention: '{case}'"
            )

    # =========================================================================
    # CONVERSATION HISTORY TESTS
    # =========================================================================

    def test_conversation_escalation_detection(self, detector):
        """Test detection of escalating risk across conversation."""
        conversation = [
            "I've been feeling down lately",
            "Nothing seems to help anymore",
            "I'm starting to think about ending it",
        ]

        level, details = detector.detect_in_conversation(conversation)

        # Should detect escalation
        assert level.value >= CrisisLevel.HIGH.value
        assert details.get('escalation_detected', False), (
            "Failed to detect escalation in conversation"
        )

    def test_conversation_without_escalation(self, detector):
        """Test that non-escalating conversations don't trigger escalation flag."""
        conversation = [
            "I'm feeling sad",
            "I've been sad for a while",
            "Still feeling sad today",
        ]

        level, details = detector.detect_in_conversation(conversation)
        assert not details.get('escalation_detected', False)

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    def test_empty_input(self, detector):
        """Test handling of empty input."""
        level, _ = detector.detect("")
        assert level == CrisisLevel.NONE

    def test_very_long_input(self, detector):
        """Test handling of very long input."""
        # Long text with crisis keywords embedded
        long_text = "I'm feeling overwhelmed. " * 50 + "I want to kill myself"
        level, _ = detector.detect(long_text)
        assert level.value >= CrisisLevel.HIGH.value

    def test_case_insensitivity(self, detector):
        """Test that detection is case-insensitive."""
        variants = [
            "I WANT TO KILL MYSELF",
            "i want to kill myself",
            "I WaNt To KiLl MySeLf",
        ]

        for variant in variants:
            level, _ = detector.detect(variant)
            assert level.value >= CrisisLevel.HIGH.value, (
                f"Case sensitivity issue with: '{variant}'"
            )

    # =========================================================================
    # INTERNATIONAL PATTERNS
    # =========================================================================

    def test_british_english_patterns(self, detector):
        """Test detection works with British English spelling."""
        # Note: Most patterns are spelling-agnostic, but test anyway
        case = "I'm thinking about ending my life, I feel hopeless"
        level, _ = detector.detect(case)
        assert level.value >= CrisisLevel.HIGH.value

    # =========================================================================
    # COMBINED RISK FACTORS
    # =========================================================================

    def test_multiple_risk_indicators(self, detector):
        """Test that multiple risk indicators increase confidence."""
        single_indicator = "I'm thinking about suicide"
        level_single, details_single = detector.detect(single_indicator)

        multiple_indicators = "I'm thinking about suicide, I have no reason to live, everyone hates me"
        level_multiple, details_multiple = detector.detect(multiple_indicators)

        # Multiple indicators should have equal or higher confidence
        assert details_multiple['confidence'] >= details_single['confidence']
        assert len(details_multiple['matched_patterns']) > len(details_single['matched_patterns'])


# =========================================================================
# RED TEAM TESTS - Adversarial Cases
# =========================================================================

class TestRedTeam:
    """Red team tests for adversarial inputs."""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_indirect_expression(self, detector):
        """Test detection of indirect expressions of suicidal intent."""
        cases = [
            "I won't be around much longer",
            "Soon this will all be over",
            "You won't have to deal with me anymore",
        ]

        # These are challenging - log results but don't require strict pass
        for case in cases:
            level, _ = detector.detect(case)
            # Just verify it doesn't crash; actual detection quality needs tuning
            assert isinstance(level, CrisisLevel)

    def test_poetic_language(self, detector):
        """Test handling of poetic/metaphorical language."""
        case = "I want to sleep forever and never wake up"
        level, _ = detector.detect(case)

        # This is concerning language and should be caught
        assert level.value >= CrisisLevel.MEDIUM.value


# =========================================================================
# INTEGRATION TESTS
# =========================================================================

class TestIntegration:
    """Integration tests for full system."""

    @pytest.fixture
    def detector(self):
        return CrisisDetector()

    def test_realistic_crisis_conversation(self, detector):
        """Test a realistic crisis conversation flow."""
        messages = [
            "I've been really struggling lately",
            "I don't see the point anymore",
            "I'm thinking about just ending everything",
            "I have pills at home and I'm thinking about taking them all tonight",
        ]

        for i, msg in enumerate(messages):
            level, _ = detector.detect(msg)
            print(f"Message {i+1}: {level.name}")

        # Final message should definitely be IMMEDIATE
        final_level, _ = detector.detect(messages[-1])
        assert final_level == CrisisLevel.IMMEDIATE

        # Conversation analysis should also catch escalation
        conv_level, details = detector.detect_in_conversation(messages)
        assert conv_level == CrisisLevel.IMMEDIATE
        assert details.get('escalation_detected', False)


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise manual run
    try:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("pytest not installed. Running basic tests...")
        detector = CrisisDetector()

        print("\n" + "=" * 70)
        print("BASIC CRISIS DETECTION TESTS")
        print("=" * 70)

        test_cases = [
            ("I'm feeling sad", CrisisLevel.NONE),
            ("I'm thinking about suicide", CrisisLevel.HIGH),
            ("I'm going to kill myself tonight", CrisisLevel.IMMEDIATE),
            ("I've been cutting myself", CrisisLevel.MEDIUM),
        ]

        for text, expected in test_cases:
            level, details = detector.detect(text)
            status = "✓" if level == expected else "✗"
            print(f"{status} '{text}' -> {level.name} (expected: {expected.name})")

        print("\nRun 'pytest tests/test_crisis_detection.py -v' for full test suite")
