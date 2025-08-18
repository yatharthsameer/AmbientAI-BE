#!/bin/bash

# Frontend Transcription Test Runner
# This script sets up and runs automated tests for the frontend transcription

echo "🤖 Frontend Transcription Test Setup"
echo "===================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Install test dependencies
echo "📦 Installing test dependencies..."
pip3 install -r test_requirements.txt

# Install Playwright browsers (for full automation)
echo "🌐 Installing Playwright browsers..."
playwright install chromium

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Choose a test method:"
echo ""
echo "1. SIMPLE TEST (Recommended)"
echo "   - You manually click Start/Stop in browser"
echo "   - Script streams audio file in 40-second chunks"
echo "   - Better Whisper context = higher accuracy"
echo "   - Easy to debug and understand"
echo ""
echo "2. FULL AUTOMATION TEST"
echo "   - Completely automated browser interaction"
echo "   - Requires more setup but fully hands-off"
echo ""

read -p "Choose test method (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "🧪 Running Simple Test..."
        echo ""
        echo "INSTRUCTIONS:"
        echo "1. Start your backend: python main.py"
        echo "2. Start your frontend: cd frontend && npm run dev"
        echo "3. Open browser to http://localhost:5173"
        echo "4. Navigate to Conversation page"
        echo "5. Click 'Start Recording'"
        echo "6. The script will then stream audio"
        echo ""
        read -p "Press Enter when ready..."
        python3 test_frontend_simple.py
        ;;
    2)
        echo ""
        echo "🧪 Running Full Automation Test..."
        echo "This will automatically start servers and control the browser"
        echo ""
        python3 test_frontend_automation.py
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ Test completed!"
echo "📋 Check the results above and in your browser's Downloads folder."
