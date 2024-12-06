## Compare Jax and Aesara

This repository was created to test that Jax and aesara compute the same values for
an optimization problem in the Covert Lab's whole cell model.

### Installation

Create a Python virtual environment with Python 3.11 or earlier. Then, install the requirements:

    pip install -r requirements

### Running

Invoke `pytest -s` to run the tests in `test_jax_aesara.py`.
The last test (`test_optimization`) can take several minutes to run
and compares the solutions of an optimization problem formulated in
three different ways:

- Simple minimization of sqaured residuals using Jax
- Multi-dimensional root finding using Jax
- Multi-dimensional root finding using Aesara

Note that `scipy.optimize.minimize` will automatically parallelize over the number
of CPU cores on your system. If you want to compare single-core timings,
set the environment variable `OMP_NUM_THREADS=1`. Additionally, you can tweak the accuracy of the minimization solution by adding the `tol` parameter.

### Benchmarks

All timings below are from an 8-core M2 Macbook Air with Python 3.11.7, SciPy 1.14.1, and Jax 0.4.36.

| Method                             | Time (sec) | RSS vs Aesara |
|------------------------------------|----------|----------------|
| Jax min (1 CPU, tol. default) | 25.6       |  +5.3e-7              |
| Jax min (8 CPU, tol. default) | 10.2       |  +5.3e-7              |
| Jax min (1 CPU, tol. 1e-7) | 31.7       |    -8.4e-8            |
| Jax min (8 CPUs, tol. 1e-7) | 13.0       |       -8.4e-8         |
| Jax min (1 CPUs, tol. 1e-8) | 38.8       |    -2.2e-7            |
| Jax min (8 CPUs, tol. 1e-8) | 15.9       |      -2.2e-7          |
| Jax root finding                   | 35.7       |        -2.1e-7        |
| Aesara root finding                | 35.7       |      0          |

**RSS vs Aesara** is the residual sum of squares (RSS) for a method
minus the RSS for the Aesara root finding method (negative is better).
