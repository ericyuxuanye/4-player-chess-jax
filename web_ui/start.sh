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
echo "  👉 http://localhost:8080"
echo "═══════════════════════════════════════════════"
echo ""
echo "💡 Tip: Use PORT environment variable to change the port:"
echo "   PORT=3000 ./start.sh"
echo ""

# Start the Flask server using flask run
export FLASK_APP=app.py
export FLASK_DEBUG=1
flask run --host=0.0.0.0 --port=8080
