#!/usr/bin/env python3
"""Test API keys for all supported LLM providers.

This script checks if API keys are configured in .env.local and validates them
by making a simple test request to each provider.

Usage:
    python scripts/test_api_keys.py           # Test all configured keys
    python scripts/test_api_keys.py openai    # Test only OpenAI
    python scripts/test_api_keys.py anthropic # Test only Anthropic
    python scripts/test_api_keys.py google    # Test only Google
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv  # noqa: E402

# Load environment variables
load_dotenv(project_root / ".env.local")


def test_openai_key(api_key: str | None = None) -> tuple[bool, str]:
    """Test OpenAI API key validity.

    Args:
        api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.

    Returns:
        Tuple of (success: bool, message: str)
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        return False, "OPENAI_API_KEY not found in environment"

    try:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, max_tokens=10)
        response = model.invoke("Say 'OK' if you can hear me.")
        return True, f"OpenAI API key is valid. Response: {response.content[:50]}"
    except Exception as e:
        return False, f"OpenAI API key validation failed: {e}"


def test_anthropic_key(api_key: str | None = None) -> tuple[bool, str]:
    """Test Anthropic API key validity.

    Args:
        api_key: Optional API key. If not provided, uses ANTHROPIC_API_KEY env var.

    Returns:
        Tuple of (success: bool, message: str)
    """
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return False, "ANTHROPIC_API_KEY not found in environment"

    try:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key, max_tokens=10)
        response = model.invoke("Say 'OK' if you can hear me.")
        return True, f"Anthropic API key is valid. Response: {response.content[:50]}"
    except Exception as e:
        return False, f"Anthropic API key validation failed: {e}"


def test_google_key(api_key: str | None = None) -> tuple[bool, str]:
    """Test Google API key validity.

    Args:
        api_key: Optional API key. If not provided, uses GOOGLE_API_KEY env var.

    Returns:
        Tuple of (success: bool, message: str)
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return False, "GOOGLE_API_KEY not found in environment"

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=api_key, max_output_tokens=10
        )
        response = model.invoke("Say 'OK' if you can hear me.")
        return True, f"Google API key is valid. Response: {response.content[:50]}"
    except Exception as e:
        return False, f"Google API key validation failed: {e}"


def test_all_keys() -> dict[str, tuple[bool, str]]:
    """Test all configured API keys.

    Returns:
        Dictionary mapping provider name to (success, message) tuple.
    """
    results = {}

    # Check which keys are configured
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    print("=" * 60)
    print("CV Warlock - API Key Validation")
    print("=" * 60)
    print()

    # Test OpenAI
    if openai_key:
        print("Testing OpenAI API key...")
        results["openai"] = test_openai_key(openai_key)
    else:
        results["openai"] = (False, "Not configured")

    # Test Anthropic
    if anthropic_key:
        print("Testing Anthropic API key...")
        results["anthropic"] = test_anthropic_key(anthropic_key)
    else:
        results["anthropic"] = (False, "Not configured")

    # Test Google
    if google_key:
        print("Testing Google API key...")
        results["google"] = test_google_key(google_key)
    else:
        results["google"] = (False, "Not configured")

    print()
    return results


def print_results(results: dict[str, tuple[bool, str]]) -> int:
    """Print test results and return exit code.

    Args:
        results: Dictionary of provider -> (success, message)

    Returns:
        0 if all configured keys are valid, 1 otherwise
    """
    print("Results:")
    print("-" * 60)

    all_configured_valid = True
    any_configured = False

    for provider, (success, message) in results.items():
        status = "PASS" if success else "FAIL"
        icon = "[OK]" if success else "[X]"

        if "Not configured" not in message:
            any_configured = True
            if not success:
                all_configured_valid = False

        print(f"  {icon} {provider.upper():12} [{status}] {message}")

    print("-" * 60)

    if not any_configured:
        print("\nNo API keys configured. Add keys to .env.local:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  GOOGLE_API_KEY=...")
        return 1

    if all_configured_valid:
        print("\nAll configured API keys are valid!")
        return 0
    else:
        print("\nSome API keys failed validation. Check the errors above.")
        return 1


def main():
    """Main entry point."""
    # Check for specific provider argument
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        if provider == "openai":
            success, message = test_openai_key()
            print(f"OpenAI: {'PASS' if success else 'FAIL'} - {message}")
            sys.exit(0 if success else 1)
        elif provider == "anthropic":
            success, message = test_anthropic_key()
            print(f"Anthropic: {'PASS' if success else 'FAIL'} - {message}")
            sys.exit(0 if success else 1)
        elif provider == "google":
            success, message = test_google_key()
            print(f"Google: {'PASS' if success else 'FAIL'} - {message}")
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown provider: {provider}")
            print("Valid options: openai, anthropic, google")
            sys.exit(1)

    # Test all keys
    results = test_all_keys()
    exit_code = print_results(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
