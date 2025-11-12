#!/bin/bash

echo "🏰 Starting 4-Player Chess Web UI..."
echo ""
echo "📦 Checking dependencies..."

# Check if dependencies are installed
python3 -c "import flask, flask_cors, jax" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r ../requirements.txt
fi

echo "✓ Dependencies OK"
echo ""
echo "🚀 Starting server..."
echo ""
echo "═══════════════════════════════════════════════"
echo "  🎮 Open your browser and go to:"
echo "  👉 http://localhost:5000"
echo "═══════════════════════════════════════════════"
echo ""

# Start the Flask server
python3 app.py
