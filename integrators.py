import numpy as np

C_TWO = (3 - np.sqrt(3)) / 6

C_THREE = 12127897 / 102017882
D_THREE = 4271554 / 14421423


def leapfrog_slow(p, q, distribution, path_len, step_size):
    p = [p_i.copy() for p_i in p]
    q = [q_i.copy() for q_i in q]

    assert(len(p) == len(q))

    for _ in range(int(path_len / step_size)):

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * dq_i / 2

        for p_i, q_i in zip(p, q):
            q_i += step_size * p_i

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * dq_i / 2

    return [-p_i for p_i in p], q


def leapfrog(p, q, distribution, path_len, step_size):
    p = [p_i.copy() for p_i in p]
    q = [q_i.copy() for q_i in q]

    assert(len(p) == len(q))

    # initial half step
    dq = distribution.negative_log_posterior_gradient(q)
    for p_i, dq_i in zip(p, dq):
        p_i += -step_size * dq_i / 2

    # Whole steps for the length of the path
    for _ in range(int(path_len / step_size) - 1):

        for p_i, q_i in zip(p, q):
            q_i += step_size * p_i

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * dq_i

    # Ending half steps
    for p_i, q_i in zip(p, q):
        q_i += step_size * p_i

    dq = distribution.negative_log_posterior_gradient(q)
    for p_i, dq_i in zip(p, dq):
        p_i += -step_size * dq_i / 2

    return [-p_i for p_i in p], q


def two_stage_sympletic(p, q, distribution, path_len, step_size):
    p = [p_i.copy() for p_i in p]
    q = [q_i.copy() for q_i in q]

    assert(len(p) == len(q))

    for _ in range(int(path_len / step_size)):

        # First Momentum
        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * C_TWO * dq_i

        # First Position Update
        for p_i, q_i in zip(p, q):
            q_i += step_size * p_i / 2

        # Second Momentum Update
        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * (1 - 2 * C_TWO) * dq_i

        # Second Position Update
        for p_i, q_i in zip(p, q):
            q_i += step_size * p_i / 2

        # Third Momentum update
        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * C_TWO * dq_i

    return [-p_i for p_i in p], q


def three_stage_symplectic(p, q, distribution, path_len, step_size):
    p = [p_i.copy() for p_i in p]
    q = [q_i.copy() for q_i in q]

    assert(len(p) == len(q))

    for _ in range(int(path_len / step_size)):

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * C_THREE * dq_i

        for p_i, q_i in zip(p, q):
            q_i += step_size * D_THREE * p_i

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * (0.5 - C_THREE) * dq_i

        for p_i, q_i in zip(p, q):
            q_i += step_size * (1 - 2 * D_THREE) * p_i

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * (0.5 - C_THREE) * dq_i

        for p_i, q_i in zip(p, q):
            q_i += step_size * D_THREE * p_i

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i, dq_i in zip(p, dq):
            p_i += -step_size * C_THREE * dq_i

    return [-p_i for p_i in p], q


def stochastic_euler_forward(
        p, q, distribution, noise_distribution, friction, path_len, step_size):
    p = [p_i.copy() for p_i in p]
    q = [q_i.copy() for q_i in q]

    n_steps = int(path_len / step_size)

    for _ in range(n_steps):

        q += step_size*p

        dq = distribution.negative_log_posterior_gradient(q)
        noise = noise_distribution.rvs(1)  # See if drawing in bulk is faster
        p += -step_size * dq - step_size*p*friction + \
            noise  # Assume that friction is a scalar for now

    return -p, q
