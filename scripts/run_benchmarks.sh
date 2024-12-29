#!/bin/bash

# Directory setup
BENCHMARK_DIR="benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BENCHMARK_DIR"

# Run speed benchmarks
echo "Running speed benchmarks..."
./build/benchmark_speed --games 100 --output "$BENCHMARK_DIR/speed_${TIMESTAMP}.csv"

# Run strength benchmarks
echo "Running strength benchmarks..."
./build/benchmark_strength --opponents "random,mcts_pure" \
                         --games 100 \
                         --output "$BENCHMARK_DIR/strength_${TIMESTAMP}.csv"

# Generate plots
python3 scripts/plot_benchmarks.py \
    --speed "$BENCHMARK_DIR/speed_${TIMESTAMP}.csv" \
    --strength "$BENCHMARK_DIR/strength_${TIMESTAMP}.csv" \
    --output "$BENCHMARK_DIR/plots_${TIMESTAMP}"

echo "Benchmarks complete. Results in $BENCHMARK_DIR" 