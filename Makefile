# Makefile for UAV Repositioning Project

# Python interpreter
PYTHON = python3

# Default target
all: install

# Create virtual environment and install dependencies
install:
	@echo "Creating virtual environment and installing dependencies..."
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Run training
train:
	@echo "Starting training..."
	. venv/bin/activate && $(PYTHON) train.py

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf venv
	rm -rf models/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Run tests (if you add tests later)
test:
	@echo "Running tests..."
	. venv/bin/activate && pytest

# Help target
help:
	@echo "Available targets:"
	@echo "  make install  - Create virtual environment and install dependencies"
	@echo "  make train    - Run the training script"
	@echo "  make clean    - Clean up generated files and virtual environment"
	@echo "  make test     - Run tests (if available)"
	@echo "  make help     - Show this help message"

.PHONY: all install train clean test help 