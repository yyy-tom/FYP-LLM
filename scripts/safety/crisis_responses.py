#!/usr/bin/env python3
"""
Crisis response templates for mental health emergencies.
These responses prioritize user safety and provide immediate resources.
"""

from typing import Dict
from .crisis_detector import CrisisLevel


class CrisisResponseGenerator:
    """
    Generate appropriate crisis responses based on risk level.

    All responses include:
    - Immediate crisis resources (988, Crisis Text Line)
    - Empathetic acknowledgment
    - Clear action steps
    - Professional help encouragement
    """

    # International crisis hotlines
    INTERNATIONAL_RESOURCES = {
        'US': {
            'suicide_hotline': '988',
            'crisis_text': 'Text HOME to 741741',
            'emergency': '911',
        },
        'UK': {
            'samaritans': '116 123',
            'crisis_text': 'Text SHOUT to 85258',
            'emergency': '999',
        },
        'Canada': {
            'crisis_line': '1-833-456-4566',
            'crisis_text': 'Text 45645',
            'emergency': '911',
        },
        'Australia': {
            'lifeline': '13 11 14',
            'suicide_callback': '1300 659 467',
            'emergency': '000',
        },
    }

    @staticmethod
    def get_immediate_response(country: str = 'US') -> str:
        """
        Response for IMMEDIATE danger (active suicidal intent).

        This response:
        - Expresses urgent concern
        - Provides immediate crisis resources prominently
        - Gives specific safety steps
        - Emphasizes that help is available 24/7
        """
        resources = CrisisResponseGenerator.INTERNATIONAL_RESOURCES.get(
            country,
            CrisisResponseGenerator.INTERNATIONAL_RESOURCES['US']
        )

        return f"""I'm very concerned about your safety right now. Please know that you don't have to face this alone, and you deserve help and support.

🆘 **IMMEDIATE HELP - Please contact now:**

**If you're in the US:**
• 988 Suicide & Crisis Lifeline: Call or text **988** (24/7, free, confidential)
• Crisis Text Line: Text **HOME** to **741741** (24/7, free)
• Emergency Services: **911**

**If you're outside the US:**
• International Crisis Lines: https://findahelpline.com/
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

**Right now - please do these things:**

1. **Call or text one of the numbers above** - Crisis counselors are standing by right now, specially trained to help people in exactly your situation
2. **Remove immediate means of harm** - Put away anything you might use to hurt yourself
3. **Don't be alone** - If possible, go to a friend/family member or public place
4. **Go to an emergency room** if you feel you cannot keep yourself safe

**Important:** The people at these crisis lines WANT to help you. They've helped thousands of people through moments like this. You don't have to give your name if you don't want to. Just call or text.

Your life has value. This moment of crisis can pass with the right support."""

    @staticmethod
    def get_high_risk_response(country: str = 'US') -> str:
        """
        Response for HIGH risk (strong suicidal ideation without immediate plan).

        This response:
        - Validates feelings
        - Provides crisis resources
        - Encourages immediate action
        - Asks engagement question
        """
        return """I hear that you're going through an incredibly difficult time, and I'm concerned about what you're sharing. These feelings can be overwhelming, but help is available right now.

**Crisis Resources Available 24/7:**
• 988 Suicide & Crisis Lifeline: **988** (call or text)
• Crisis Text Line: Text **HOME** to **741741**
• For international support: https://findahelpline.com/

**Why reach out to crisis services?**
- They're specifically trained for these exact situations
- Available 24/7, free and confidential
- They can help you work through these feelings in a safe way
- No pressure, no judgment - just support

**Please consider right now:**
1. Calling or texting 988 - just to talk to someone who understands
2. If you're not ready to call, try texting the Crisis Text Line
3. Telling someone you trust about how you're feeling
4. Making an appointment with a mental health professional if you haven't already

While I want to support you, trained crisis counselors can provide specialized help that I can't. Many people who felt exactly like you do now have found relief through reaching out.

Would you be willing to contact one of these resources? Or is there something specific that's making you hesitate?"""

    @staticmethod
    def get_medium_risk_response(country: str = 'US') -> str:
        """
        Response for MEDIUM risk (self-harm thoughts, hopelessness).

        This response:
        - Acknowledges courage in sharing
        - Provides resources (less urgently than HIGH/IMMEDIATE)
        - Offers coping strategies
        - Encourages professional help
        """
        return """Thank you for sharing these difficult feelings with me. It takes courage to talk about this, and I'm glad you did.

**Support Resources:**
• 988 Suicide & Crisis Lifeline: **988** (24/7, call or text)
• Crisis Text Line: Text **HOME** to **741741** (24/7)

If you're having thoughts of harming yourself, please consider reaching out to a crisis counselor or mental health professional. They have specialized training to help with exactly these kinds of feelings.

**Some strategies that might help right now:**

1. **Grounding technique** - Try the 5-4-3-2-1 method:
   - Name 5 things you can see
   - 4 things you can touch
   - 3 things you can hear
   - 2 things you can smell
   - 1 thing you can taste

2. **Reach out** - Text or call someone you trust, even just to chat about something else

3. **Delay and distract** - If you're having urges to self-harm, try waiting 15 minutes while doing something engaging (puzzle, video, walk)

4. **Safe space** - Go to a comfortable, safe place where you feel more secure

**Remember:**
- Feelings, even intense ones, are temporary
- You deserve support and care
- Professional help can make a real difference

Would you like to talk about what's been contributing to these feelings? Or would you prefer suggestions for next steps to get support?"""

    @staticmethod
    def get_low_risk_response() -> str:
        """
        Response for LOW risk (general distress, manageable symptoms).

        This response:
        - Validates feelings
        - Provides general mental health resources
        - Offers to listen
        - Suggests self-care
        """
        return """I hear that you're going through a difficult time. It's important to acknowledge these feelings and take care of your mental health.

**If things feel overwhelming:**
• 988 Suicide & Crisis Lifeline: **988** (available 24/7)
• Crisis Text Line: Text **HOME** to **741741**

**Some things that might help:**

1. **Self-care basics** - Sleep, nutrition, exercise, and connection with others
2. **Talk to someone** - A friend, family member, or counselor
3. **Professional support** - Consider seeing a therapist if you're not already
4. **Stress management** - Deep breathing, meditation, or activities you enjoy

**Resources for ongoing support:**
- SAMHSA National Helpline: 1-800-662-4357 (treatment referral service)
- Psychology Today therapist finder: psychologytoday.com/us/therapists
- BetterHelp, Talkspace (online therapy platforms)

Would you like to talk more about what you're experiencing? I'm here to listen and offer support."""

    @staticmethod
    def get_system_disclaimer() -> str:
        """
        General system disclaimer shown at startup.
        """
        return """
╔════════════════════════════════════════════════════════════════╗
║              MENTAL HEALTH SUPPORT ASSISTANT                   ║
╠════════════════════════════════════════════════════════════════╣
║  This is an AI assistant providing mental health information   ║
║  and support. It is NOT a substitute for professional mental   ║
║  health care or emergency services.                            ║
║                                                                ║
║  🆘 IF YOU'RE IN CRISIS OR EMERGENCY:                          ║
║                                                                ║
║     • US: Call or text 988 (Suicide & Crisis Lifeline)        ║
║     • US: Text HOME to 741741 (Crisis Text Line)              ║
║     • Emergency: 911 (US) or your local emergency number      ║
║     • International: https://findahelpline.com/               ║
║                                                                ║
║  For ongoing mental health support, please consult with a      ║
║  licensed mental health professional.                          ║
║                                                                ║
║  This tool is designed to provide information and support,     ║
║  but it cannot replace professional diagnosis, treatment, or   ║
║  crisis intervention.                                          ║
╚════════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def get_response_for_level(crisis_level: CrisisLevel, country: str = 'US') -> str:
        """
        Get appropriate response for a given crisis level.

        Args:
            crisis_level: Detected crisis level
            country: Country code for localized resources (default: 'US')

        Returns:
            Appropriate crisis response string
        """
        if crisis_level == CrisisLevel.IMMEDIATE:
            return CrisisResponseGenerator.get_immediate_response(country)
        elif crisis_level == CrisisLevel.HIGH:
            return CrisisResponseGenerator.get_high_risk_response(country)
        elif crisis_level == CrisisLevel.MEDIUM:
            return CrisisResponseGenerator.get_medium_risk_response(country)
        elif crisis_level == CrisisLevel.LOW:
            return CrisisResponseGenerator.get_low_risk_response()
        else:
            return ""  # No crisis response needed

    @staticmethod
    def get_footer_banner() -> str:
        """
        Persistent footer shown in UI during sessions.
        """
        return """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🆘 In Crisis? → Call/Text 988 (US) | Text HOME to 741741
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Example usage
if __name__ == "__main__":
    generator = CrisisResponseGenerator()

    print(generator.get_system_disclaimer())
    print("\n" + "=" * 70)
    print("Example IMMEDIATE Response:")
    print("=" * 70)
    print(generator.get_immediate_response())

    print("\n" + "=" * 70)
    print("Example HIGH Risk Response:")
    print("=" * 70)
    print(generator.get_high_risk_response())

    print("\n" + generator.get_footer_banner())
