# RGDBEK
Randomized Greedy Double Block Extended Kaczmarz Algorithm [[arXiv](https://arxiv.org/abs/2509.19267)] [[Wiley](https://doi.org/10.1002/nla.70102)]

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

## BibTeX
```bibtex
@article {MR5090111,
    AUTHOR = {Panchal, Aneesh and Behera, Ratikanta},
     TITLE = {R{GDBEK}: {R}andomized {G}reedy {D}ouble {B}lock {E}xtended
              {K}aczmarz {A}lgorithm {W}ith {H}ybrid {P}arallel
              {I}mplementation and {A}pplications},
   JOURNAL = {Numer. Linear Algebra Appl.},
  FJOURNAL = {Numerical Linear Algebra with Applications},
    VOLUME = {33},
      YEAR = {2026},
    NUMBER = {3},
     PAGES = {Paper No. e70102},
      ISSN = {1070-5325,1099-1506},
   MRCLASS = {65F10 (65F50 65Y05)},
  MRNUMBER = {5090111},
       DOI = {10.1002/nla.70102},
       URL = {https://doi.org/10.1002/nla.70102},
}
```
