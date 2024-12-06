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
