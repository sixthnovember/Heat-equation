#!/bin/bash
#SBATCH -N 6
#SBATCH --ntasks 96
mpirun ./a2 --m 1152 --n 1152 --epsilon 0.001 --max-iterations 1000
