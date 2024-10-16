#!/bin/bash

# Check if a number was passed as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <number>"
  exit 1
fi

# Assign the input number to a variable
NUMBER=$1

# Run the command with the input number substituted
xtpen ~/WORK/research/mars2024/mltrSPSLAKE/T/dsyT2-r${NUMBER}.vpl ../../../512512/clutter/outputs/mdl-${NUMBER}.vpl ../../../512512/clutter2/outputs/mdl-${NUMBER}.vpl mdl-${NUMBER}.vpl

