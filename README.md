# RGDBEK
Randomized Greedy Double Block Extended Kaczmarz Algorithm [arXiv](https://arxiv.org/abs/2509.19267)

## Run Sequential
Sequential experiments are performed on MATLAB R2025a.
``` bash
main
```

## Run Parallel
Parallel experiments are performed in Python 3.10.13 using CUDA 12.4.131 and MPI 4.0.2.
``` bash
pip install -r requirements.txt
chmod +x run_mpi.sh
./run_mpi.sh
```

## Run FEM Applications
FEM Applications are performed using sequential RGDBEK on Python 3.11.0.
``` bash
pip install -r requirements.txt
python poisson.py
python helmholtz.py
```

Cite this work as,
