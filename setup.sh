#!/bin/bash
# Setup script for OpenScope RL Training

echo "================================================"
echo "OpenScope RL Training System Setup"
echo "================================================"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install chromium

# Create directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p evaluation_results
mkdir -p training_plots

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the openScope game server in another terminal:"
echo "   cd .."
echo "   npm install"
echo "   npm run build"
echo "   npm run start"
echo ""
echo "3. Start training:"
echo "   python train.py"
echo ""
echo "For more information, see README.md"

