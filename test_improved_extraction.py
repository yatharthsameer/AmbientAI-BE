"""
Test script for improved hybrid medical information extraction with final verification.
"""
import asyncio
import logging
from services.gemini_service import GeminiService
from services.distilbert_service import DistilBERTService
from services.hybrid_extraction_service import HybridExtractionService
from services.final_verification_service import FinalVerificationService
from config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample conversation text
SAMPLE_CONVERSATION = """
Sarah: Good morning, Mrs. Rodriguez! It's Sarah from HomeCare Health. How're you feeling today?
Mrs. Rodriguez: Hello, Sarah. Come in. I'm sore, but better than yesterday. This hip's still a bother.
Sarah: I bet. Let's get you to your chair. Mind if I wash my hands first? (Washes hands at sink.) Alright, today we'll check your vitals, review meds, and maybe try some exercises. Sound okay?
Mrs. Rodriguez: Fine by me. Want some tea? I made a pot.
Sarah: Sweet of you, but I'll pass—let's focus on you. Pain level, 1 to 10?
Mrs. Rodriguez: About a 5. Hurts more when I walk.
(Sarah notes this.)
Sarah: Let's start with blood pressure and temperature. Sleeve up, please? (Takes BP: 128/82, temp: 98.6°F.) BP's a bit high, but not bad post-surgery. Blood sugar this morning?
Mrs. Rodriguez: 140 before breakfast. Took my insulin like you showed me.
Sarah: Nice work! Let's check it now. (Does glucometer test: 135 mg/dL.) Steady—good job with your diet. Any dizziness or falls?
Mrs. Rodriguez: No falls, thank God. But I'm tired a lot. And lonely—my daughter's far away.
Sarah: Recovery can feel isolating. We'll talk more about that. Let's see your incision. Lift your gown? (Inspects hip.) Looks good—no redness. Keep using that antibiotic cream twice a day, okay?
Mrs. Rodriguez: Will do. You're gentle, Sarah.
(Sarah documents, washes hands.)
Sarah: Let's go over meds. Show me your pill organizer? (Mrs. Rodriguez hands it over.) Oxycodone every 6 hours as needed? How many yesterday?
Mrs. Rodriguez: Two in the morning, one at night. Helps, but I'm constipated.
Sarah: Common side effect. Try prunes or oatmeal, and lots of water. Diabetes meds: Metformin twice a day, insulin at meals. Any low blood sugar?
Mrs. Rodriguez: Once last week—felt shaky, ate candy like you said.
Sarah: Smart! Remember 15 grams of carbs, wait 15 minutes, recheck. Here's a handout. (Gives sheet.) Change your dressing daily, call if it's warm or oozing. Questions?
Mrs. Rodriguez: The exercises—the leg lifts hurt.
Sarah: Let's ease into those. Show me how you do them? (Guides Mrs. Rodriguez through demo.) Perfect—10 reps, three times a day.
(Pause for practice.)
Sarah: Feeling okay? Let's try walking to the kitchen with your walker. (They walk slowly.) Nice posture—keep that up.
"""

# Sample questions
SAMPLE_QUESTIONS = [
    {"id": "patient_name", "question": "What is the patient's name?"},
    {"id": "pain_level", "question": "What is the patient's pain level?"},
    {"id": "medications", "question": "What medications is the patient taking?"},
    {"id": "vital_signs", "question": "What vital signs were checked?"},
    {"id": "blood_pressure", "question": "What was the patient's blood pressure?"},
    {"id": "temperature", "question": "What was the patient's temperature?"},
    {"id": "blood_sugar", "question": "What was the patient's blood sugar level?"},
    {"id": "exercises", "question": "What exercises were demonstrated or performed?"},
    {"id": "chief_complaint", "question": "What is the patient's main problem or complaint?"},
    {"id": "allergies", "question": "Does the patient have any allergies?"}
]

async def test_improved_extraction():
    """Test the improved hybrid extraction system with final verification."""
    try:
        logger.info("🧪 Testing improved hybrid medical information extraction with final verification...")
        
        # Get settings
        settings = get_settings()
        
        # Initialize services
        gemini_service = None
        distilbert_service = None
        
        if settings.gemini_settings.api_key:
            gemini_service = GeminiService(settings.gemini_settings)
            logger.info("✅ Gemini service initialized")
        else:
            logger.warning("⚠️ No Gemini API key found")
        
        try:
            distilbert_service = DistilBERTService()
            logger.info("✅ DistilBERT service initialized")
        except Exception as e:
            logger.error(f"❌ DistilBERT service failed: {e}")
            return
        
        if not gemini_service:
            logger.error("❌ Cannot test hybrid service without Gemini")
            return
        
        # Initialize hybrid service
        hybrid_service = HybridExtractionService(gemini_service, distilbert_service)
        logger.info("✅ Hybrid extraction service with verification initialized")
        
        # Test extraction
        logger.info("🔍 Starting improved hybrid extraction with verification...")
        start_time = asyncio.get_event_loop().time()
        
        results = await hybrid_service.extract_medical_info(
            conversation_text=SAMPLE_CONVERSATION,
            questions=SAMPLE_QUESTIONS
        )
        
        extraction_time = asyncio.get_event_loop().time() - start_time
        
        # Display results
        logger.info("\n📊 IMPROVED EXTRACTION RESULTS:")
        logger.info("=" * 60)
        
        quality_score = results.get('quality_score', 0.0)
        extraction_method = results.get('extraction_method', 'unknown')
        extraction_timestamp = results.get('extraction_timestamp', 'unknown')
        conversation_length = results.get('conversation_length', 0)
        
        logger.info(f"🎯 Quality Score: {quality_score:.2f}")
        logger.info(f"🔧 Extraction Method: {extraction_method}")
        logger.info(f"⏰ Extraction Timestamp: {extraction_timestamp}")
        logger.info(f"📏 Conversation Length: {conversation_length} characters")
        logger.info("")
        
        for question_data in SAMPLE_QUESTIONS:
            question_id = question_data['id']
            question_text = question_data['question']
            
            if question_id in results:
                answer = results[question_id]
                logger.info(f"❓ {question_id}:")
                logger.info(f"   Question: {question_text}")
                logger.info(f"   Answer: {answer}")
                logger.info("")
            else:
                logger.info(f"❓ {question_id}: No answer found")
        
        # Summary statistics
        total_questions = len(SAMPLE_QUESTIONS)
        answered_questions = sum(
            1 for question_id in SAMPLE_QUESTIONS
            if question_id in results and results[question_id] != "Information not available in conversation"
        )
        
        logger.info("📈 SUMMARY:")
        logger.info(f"   Total questions: {total_questions}")
        logger.info(f"   Questions answered: {answered_questions}")
        logger.info(f"   Success rate: {(answered_questions/total_questions)*100:.1f}%")
        logger.info(f"   Processing time: {extraction_time:.2f} seconds")
        logger.info(f"   Quality score: {quality_score:.2f}")
        
        # Test final verification service separately
        logger.info("\n🔍 Testing Final Verification Service separately...")
        final_verifier = FinalVerificationService()
        
        # Create some test data
        test_results = {
            "patient_name": "Mrs. Rodriguez",
            "pain_level": "What is the patient's pain level on a scale of 1 to 10? Sarah: Good morning...",
            "medications": "insulin",
            "blood_pressure": "128/82",
            "temperature": "98.6"
        }
        
        verified_test = final_verifier.verify_and_clean_results(
            test_results, SAMPLE_CONVERSATION, SAMPLE_QUESTIONS
        )
        
        logger.info("📋 Verification Test Results:")
        for question_id, answer in verified_test.items():
            if question_id not in ['extraction_timestamp', 'conversation_length']:
                logger.info(f"   {question_id}: {answer}")
        
        test_quality = final_verifier.get_quality_score(verified_test)
        logger.info(f"   Test Quality Score: {test_quality:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_improved_extraction())

