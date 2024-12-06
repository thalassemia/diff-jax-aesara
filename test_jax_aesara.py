import numpy as np
import pytest
from numpy.testing import assert_allclose

import jax
# Can comment out to test Jax with 32-bit floats
jax.config.update("jax_enable_x64", True)
from jax import jacfwd
import jax.numpy as jnp
import aesara.tensor as T
from aesara import function, gradient
import scipy.optimize
import time


def jax_simple(vMax, rnaConc, kDeg, isEndoRnase, alpha):
    def residual_f(km):
        return vMax / km / kDeg / (1 + jnp.sum(rnaConc / km)) - 1

    def residual_aux_f(km):
        return vMax * rnaConc / km / (1 + jnp.sum(rnaConc / km)) - kDeg * rnaConc

    def L(log_km):
        km = jnp.exp(log_km)
        residual_squared = residual_f(km) ** 2

        # Loss function
        return jnp.sum(residual_squared)

    def Lp(km):
        return jacfwd(L)(km)

    return (
        L,
        Lp,
        residual_f,
        residual_aux_f,
    )


def jax_faithful(vMax, rnaConc, kDeg, isEndoRnase, alpha):
    def residual_f(km):
        return vMax / km / kDeg / (1 + (rnaConc / km).sum()) - 1

    def residual_aux_f(km):
        return vMax * rnaConc / km / (1 + (rnaConc / km).sum()) - kDeg * rnaConc

    def regularizationNegativeNumbers_f(km):
        return (1 - km / jnp.abs(km)).sum() / rnaConc.size

    def L(km):
        residual = residual_f(km)
        regularizationNegativeNumbers = regularizationNegativeNumbers_f(km)

        # Regularization for endoRNAse Km values
        regularizationEndoR = (isEndoRnase * jnp.abs(residual)).sum()

        # Multi-objective regularization
        WFendoR = 0.1  # weighting factor to protect Km optimized of EndoRNases
        regularization = regularizationNegativeNumbers + WFendoR * regularizationEndoR

        # Loss function
        return jnp.log(jnp.exp(residual) + jnp.exp(alpha * regularization)) - jnp.log(2)

    def Lp(km):
        return jacfwd(L)(km)

    return (
        L,
        Lp,
        residual_f,
        residual_aux_f,
    )


def aesara(vMax, rnaConc, kDeg, isEndoRnase, alpha):
    N = rnaConc.size
    km = T.dvector()

    # Residuals of non-linear optimization
    residual = (vMax / km / kDeg) / (1 + (rnaConc / km).sum()) - np.ones(N)
    residual_aux = (vMax * rnaConc / km) / (1 + (rnaConc / km).sum()) - (kDeg * rnaConc)

    # Counting negative Km's (first regularization term)
    regularizationNegativeNumbers = (np.ones(N) - km / np.abs(km)).sum() / N

    # Penalties for EndoR Km's, which might be potentially nonf-fitted
    regularizationEndoR = (isEndoRnase * np.abs(residual)).sum()

    # Multi objective-based regularization
    WFendoR = 0.1  # weighting factor to protect Km optimized of EndoRNases
    regularization = regularizationNegativeNumbers + (WFendoR * regularizationEndoR)

    # Loss function
    LossFunction = T.log(T.exp(residual) + T.exp(alpha * regularization)) - T.log(2)

    L = function([km], LossFunction)
    R = function([km], residual)
    Lp = function([km], gradient.jacobian(LossFunction, km))
    R_aux = function([km], residual_aux)

    return L, Lp, R, R_aux


@pytest.fixture
def test_data():
    return (
        3.4492542873610095e-07,
        np.loadtxt("rna_conc.txt"),
        np.loadtxt("kdeg.txt"),
        np.loadtxt("is_endornase.txt"),
        0.5,
    )


@pytest.fixture
def km_counts():
    return np.loadtxt("km_counts.txt")


def test_loss_function_equivalence(test_data, km_counts):
    """Test if loss functions produce equivalent results"""
    # Get loss functions
    jax_loss, _, _, _ = jax_faithful(*test_data)
    aesara_loss, _, _, _ = aesara(*test_data)

    # Compute outputs
    jax_output = jax_loss(km_counts)
    aesara_output = aesara_loss(km_counts)

    # Compare with tolerance
    assert_allclose(jax_output, aesara_output, rtol=1e-5, atol=1e-8)


def test_jacobian_equivalence(test_data, km_counts):
    """Test if Jacobians produce equivalent results"""
    # Get Jacobian functions
    _, jax_jacobian, _, _ = jax_faithful(*test_data)
    _, aesara_jacobian, _, _ = aesara(*test_data)

    # Compute Jacobians
    jax_output = jax_jacobian(km_counts)
    aesara_output = aesara_jacobian(km_counts)

    # Compare with tolerance
    assert_allclose(jax_output, aesara_output, rtol=1e-5, atol=1e-8)


def test_edge_cases(test_data, km_counts):
    """Test behavior with edge cases"""
    # Create edge case inputs
    zero_counts = np.zeros_like(km_counts)
    small_counts = np.full_like(km_counts, 1e-10)
    large_counts = np.full_like(km_counts, 1e10)

    edge_cases = [zero_counts, small_counts, large_counts]

    for case in edge_cases:
        # Get functions
        jax_loss, jax_jacobian, _, _ = jax_faithful(*test_data)
        aesara_loss, aesara_jacobian, _, _ = aesara(*test_data)

        # Test loss functions
        assert_allclose(jax_loss(case), aesara_loss(case), rtol=1e-5, atol=1e-8)

        # Test Jacobians
        assert_allclose(jax_jacobian(case), aesara_jacobian(case), rtol=1e-5, atol=1e-8)


def test_optimization(test_data, km_counts):
    # Simple Jax
    start_time = time.time()
    jax_loss, jax_jacobian, _, _ = jax_simple(*test_data)
    jax_simple_sol = scipy.optimize.minimize(
        jax_loss, np.log(km_counts), jac=jax_jacobian
    )
    jax_simple_sol.x = np.exp(jax_simple_sol.x)
    print("Jax simple finished in: ", time.time() - start_time)

    # Faithful Jax
    start_time = time.time()
    jax_f_loss, jax_f_jacobian, _, _ = jax_faithful(*test_data)
    jax_faithful_sol = scipy.optimize.root(jax_f_loss, km_counts, jac=jax_f_jacobian)
    print("Jax faithful finished in: ", time.time() - start_time)

    # Aesara
    start_time = time.time()
    a_loss, a_jacobian, a_res, _ = aesara(*test_data)
    aesara_sol = scipy.optimize.root(a_loss, km_counts, jac=a_jacobian)
    print("Aesara finished in: ", time.time() - start_time)

    # Compare solutions
    solutions = {
        "Jax simple": jax_simple_sol.x,
        "Jax faithful": jax_faithful_sol.x,
        "Aesara": aesara_sol.x,
    }
    combos = [
        ("Jax simple", "Jax faithful"),
        ("Jax simple", "Aesara"),
        ("Jax faithful", "Aesara"),
    ]
    for function_one, function_two in combos:
        function_one_res = a_res(solutions[function_one])
        function_two_res = a_res(solutions[function_two])
        np.testing.assert_allclose(function_one_res, function_two_res, atol=1e-6)
        diff = np.max(np.abs(function_one_res - function_two_res))
        print(f"{function_one} - {function_two} max abs. diff. in residuals: {diff}")
        norm_diff = np.linalg.norm(function_one_res) - np.linalg.norm(function_two_res)
        print(f"{function_one} - {function_two} 2-norm diff. in residuals: {norm_diff}")
