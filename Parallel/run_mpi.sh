#!/bin/bash
for i in {1..10}
do
    mpiexec -n 1 python distrib.py
done
