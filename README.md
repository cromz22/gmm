# Algorithms for Gaussian mixture models

## Environmental setup
```
python3 -m venv venv  # >= python3.6
. venv/bin/activate
pip install numpy pandas scipy
```

## Usage

### EM algorithm
```
python src/em.py data/x.csv data/z.csv data/params.dat
```

### Variational Bayes algorithm
```
python src/vb.py data/x.csv data/z.csv data/params.dat
```
