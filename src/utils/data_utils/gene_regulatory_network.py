import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

SEED = 42


def generate_parameters():
    rng = np.random.default_rng(SEED)
    
    num_genes = 200

    A = rng.binomial(1, 0.03, size=(num_genes, num_genes))
    A[np.diag_indices(len(A))] = 0

    edge = np.argwhere(A != 0)
    
    activating = rng.binomial(1, 0.5, len(edge)).astype(bool)
    
    activating_edge = edge[activating]
    repressing_edge = edge[~activating]

    gamma_a = np.zeros_like(A).astype(float)
    gamma_r = np.zeros_like(A).astype(float)
    gamma_a[*activating_edge.T] = rng.uniform(0.5, 1.5, size=len(activating_edge))
    gamma_r[*repressing_edge.T] = rng.uniform(0.5, 1.5, size=len(repressing_edge))
    K = rng.uniform(0.25, 0.75, size=(num_genes, num_genes))

    return A, gamma_a, gamma_r, K


def step(c, dt, gamma_a, gamma_r, K, beta, inv_omega, key, n):
    H = lambda x: x / (1 + x)

    def _regulation_term(c, gamma_a, gamma_r, K, n):
        activation_strength = jnp.where(
            gamma_a != 0, 
            gamma_a * (c**n / (K**n + c**n)), 
            0)
        num_activations = jnp.where(
            activation_strength != 0, 
            1, 
            0).sum(axis=1)
        repression_strength = jnp.where(
            gamma_r != 0, 
            gamma_r * (K**n / (K**n + c**n)), 
            0)
        num_repressions = jnp.where(repression_strength != 0, 1, 0).sum(axis=1)
        return activation_strength.sum(axis=1) + repression_strength.sum(axis=1)
                                                

    def _stochastic_term(regulation_strength, dt, inv_omega):
        return jnp.sqrt(2 * inv_omega * regulation_strength * dt) * \
               jrandom.normal(key, shape=(len(c),))

    production = _regulation_term(c, gamma_a, gamma_r, K, n)
    degradation = beta * c
    
    forcing = (production - degradation) * dt
    noise = _stochastic_term(jnp.abs(production) + jnp.abs(degradation), dt, inv_omega)
    c_new = c + forcing + noise
    
    return c_new