#!/bin/bash

iterations=$1

echo $iterations > tmp.input

gfortran sierpinski.F90 -o sierpinski_gen.out

./sierpinski_gen.out < tmp.input

python3 visualize_sierpinski.py

python3 hamiltonian_sierpinski.py


