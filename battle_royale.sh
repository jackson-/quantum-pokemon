#!/bin/bash

# Ensure we have the base ROM
if [ ! -f "PokemonRed.gb" ]; then
    echo "Error: PokemonRed.gb not found!"
    exit 1
fi

echo "Setting up Battle Royale: Classical vs Quantum"

# Create independent ROM copies to ensure separate save files (.sav)
cp PokemonRed.gb PokemonRed_Classical.gb
cp PokemonRed.gb PokemonRed_Quantum.gb

echo "Launching Classical Agent..."
# Run in background
source .venv/bin/activate && python3 watch_classical.py PokemonRed_Classical.gb &
CLASSICAL_PID=$!

echo "Launching Quantum Agent..."
# Run in background
source .venv/bin/activate && python3 watch_quantum.py PokemonRed_Quantum.gb &
QUANTUM_PID=$!

echo "Both agents running!"
echo "Classical PID: $CLASSICAL_PID"
echo "Quantum PID: $QUANTUM_PID"
echo "Press Ctrl+C to stop both."

# Wait for both to finish or user interrupt
trap "kill $CLASSICAL_PID $QUANTUM_PID; exit" SIGINT
wait
