#!/usr/bin/env python3
"""
Simple test script for Gemini integration.
Run this to test the Gemini service without the full API.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.gemini_service import GeminiService
from config import GeminiSettings


async def test_gemini_basic():
    """Test basic Gemini functionality."""
    print("🧪 Testing Gemini Basic Functionality")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Please set it in your environment.")
        print("   export GEMINI_API_KEY=your_api_key_here")
        return False
    
    try:
        # Initialize service
        settings = GeminiSettings(api_key=api_key)
        gemini = GeminiService(settings)
        
        # Test basic content generation
        print("📝 Testing basic content generation...")
        response = await gemini.generate_content(
            prompt="Explain how AI works in simple terms",
            temperature=0.3,
            max_tokens=200
        )
        
        print(f"✅ Response received from {response.model}")
        print(f"📊 Finish reason: {response.finish_reason}")
        print(f"💬 Response text:\n{response.text}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


async def test_gemini_medical():
    """Test medical information extraction."""
    print("🏥 Testing Gemini Medical Information Extraction")
    print("=" * 50)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set.")
        return False
    
    try:
        settings = GeminiSettings(api_key=api_key)
        gemini = GeminiService(settings)
        
        # Sample medical conversation
        conversation = """
        Nurse: Good morning, Mr. Johnson. How are you feeling today?
        Patient: I'm not feeling well. I have a headache and fever.
        Nurse: I see. When did these symptoms start?
        Patient: Yesterday evening, around 8 PM.
        Nurse: Have you taken any medication?
        Patient: Yes, I took some Tylenol but it didn't help much.
        Nurse: Any other symptoms like nausea or dizziness?
        Patient: A little bit of nausea, yes.
        """
        
        # Sample medical questions
        questions = [
            {"id": "patient_name", "question": "What is the patient's name?"},
            {"id": "chief_complaint", "question": "What is the patient's chief complaint?"},
            {"id": "symptoms", "question": "What symptoms is the patient experiencing?"},
            {"id": "medications", "question": "What medications has the patient taken?"},
            {"id": "symptom_onset", "question": "When did the symptoms start?"}
        ]
        
        print("📋 Extracting medical information...")
        result = await gemini.extract_medical_info(conversation, questions)
        
        print("✅ Medical extraction completed")
        print("📊 Extracted information:")
        for key, value in result.items():
            print(f"   {key}: {value}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Medical test failed: {e}")
        return False


async def test_gemini_summary():
    """Test conversation summarization."""
    print("📝 Testing Gemini Conversation Summarization")
    print("=" * 50)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set.")
        return False
    
    try:
        settings = GeminiSettings(api_key=api_key)
        gemini = GeminiService(settings)
        
        # Sample conversation
        conversation = """
        Doctor: Hello Sarah, how are you feeling today?
        Sarah: I'm still having those chest pains, doctor.
        Doctor: Can you describe the pain for me?
        Sarah: It's like a tightness, and it gets worse when I walk.
        Doctor: How long does the pain last?
        Sarah: Usually about 5-10 minutes, then it goes away.
        Doctor: Have you noticed any other symptoms?
        Sarah: Sometimes I feel short of breath too.
        Doctor: I think we should run some tests to be safe.
        Sarah: What kind of tests?
        Doctor: An EKG and maybe a stress test to check your heart.
        """
        
        print("📋 Generating medical summary...")
        summary = await gemini.summarize_conversation(
            conversation, 
            summary_type="medical"
        )
        
        print("✅ Summary generated")
        print(f"📄 Summary:\n{summary}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Summary test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Gemini Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_gemini_basic),
        ("Medical Extraction", test_gemini_medical),
        ("Conversation Summary", test_gemini_summary)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
