import numpy as np

EPS = np.finfo('float64').tiny


def cg(f_ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """Demmel p 312."""
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose:
        print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))
    return x


def preconditioned_cg(f_ax,
                      f_minvx,
                      b,
                      cg_iters=10,
                      callback=None,
                      verbose=False,
                      residual_tol=1e-10):
    """Demmel p 318."""
    x = np.zeros_like(b)
    r = b.copy()
    p = f_minvx(b)
    y = p
    ydotr = y.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x, f_ax)
        if verbose:
            print(fmtstr % (i, ydotr, np.linalg.norm(x)))
        z = f_ax(p)
        v = ydotr / p.dot(z)
        x += v * p
        r -= v * z
        y = f_minvx(r)
        newydotr = y.dot(r)
        mu = newydotr / ydotr
        p = y + mu * p

        ydotr = newydotr

        if ydotr < residual_tol:
            break

    if verbose:
        print(fmtstr % (cg_iters, ydotr, np.linalg.norm(x)))

    return x


def test_cg():
    a = np.random.randn(5, 5)
    a = a.T.dot(a)
    b = np.random.randn(5)
    x = cg(lambda x: a.dot(x), b, cg_iters=5, verbose=True)
    assert np.allclose(a.dot(x), b)

    x = preconditioned_cg(
        lambda x: a.dot(x),
        lambda x: np.linalg.solve(a, x),
        b,
        cg_iters=5,
        verbose=True)  # pylint: disable=W0108
    assert np.allclose(a.dot(x), b)

    x = preconditioned_cg(
        lambda x: a.dot(x),
        lambda x: x / np.diag(a),
        b,
        cg_iters=5,
        verbose=True)  # pylint: disable=W0108
    assert np.allclose(a.dot(x), b)


def lanczos(f_ax, b, k):
    """Run Lanczos algorithm.

    It generate an orthogonal basis for the Krylov subspace b, Ab, A^2b, ...
    as well as the upper hessenberg matrix T = Q^T A Q

    from Demmel ch 6
    """
    assert k > 1

    alphas = []
    betas = []
    qs = []

    q = b / np.linalg.norm(b)
    beta = 0
    qm = np.zeros_like(b)
    for j in range(k):
        qs.append(q)

        z = f_ax(q)

        alpha = q.dot(z)
        alphas.append(alpha)
        z -= alpha * q + beta * qm

        beta = np.linalg.norm(z)
        betas.append(beta)

        print("beta", beta)
        if beta < 1e-9:
            print("lanczos: early after %i/%i dimensions" % (j + 1, k))
            break
        else:
            qm = q
            q = z / beta

    return np.array(qs, 'float64').T, np.array(alphas, 'float64'), np.array(
        betas[:-1], 'float64')


def lanczos2(f_ax, b, k, residual_thresh=1e-9):
    """Run Lanczos algorithm.

    It generates an orthogonal basis for the Krylov subspace b, Ab, A^2b, ...
    as well as the upper hessenberg matrix T = Q^T A Q

    from Demmel ch 6
    """
    b = b.astype('float64')
    assert k > 1
    h = np.zeros((k, k))
    qs = []

    q = b / np.linalg.norm(b)
    beta = 0

    for j in range(k):
        qs.append(q)

        z = f_ax(q.astype('float64')).astype('float64')
        for (i, q) in enumerate(qs):
            h[j, i] = h[i, j] = h = q.dot(z)
            z -= h * q

        beta = np.linalg.norm(z)
        if beta < residual_thresh:
            print("lanczos2: stopping early after %i/%i dimensions residual "
                  "%f < %f" % (j + 1, k, beta, residual_thresh))
            break
        else:
            q = z / beta

    return np.array(qs).T, h[:len(qs), :len(qs)]


def make_tridiagonal(alphas, betas):
    assert len(alphas) == len(betas) + 1
    n = alphas.size
    out = np.zeros((n, n), 'float64')
    out.flat[0:n**2:n + 1] = alphas
    out.flat[1:n**2 - n:n + 1] = betas
    out.flat[n:n**2 - 1:n + 1] = betas
    return out


def tridiagonal_eigenvalues(alphas, betas):
    t = make_tridiagonal(alphas, betas)
    return np.linalg.eigvalsh(t)


def test_lanczos():
    np.set_printoptions(precision=4)

    a = np.random.randn(5, 5)
    a = a.T.dot(a)
    b = np.random.randn(5)

    def f_ax(x):
        a.dot(x)

    q, alphas, betas = lanczos(f_ax, b, 10)
    h = make_tridiagonal(alphas, betas)
    assert np.allclose(q.T.dot(a).dot(q), h)
    assert np.allclose(q.dot(h).dot(q.T), a)
    assert np.allclose(np.linalg.eigvalsh(h), np.linalg.eigvalsh(a))

    q, h1 = lanczos2(f_ax, b, 10)
    assert np.allclose(h, h1, atol=1e-6)

    print("ritz eigvals:")
    for i in range(1, 6):
        qi = q[:, :i]
        hi = qi.T.dot(a).dot(qi)
        print(np.linalg.eigvalsh(hi)[::-1])
    print("true eigvals:")
    print(np.linalg.eigvalsh(a)[::-1])

    print("lanczos on ill-conditioned problem")
    a = np.diag(10**np.arange(5))
    q, h1 = lanczos2(f_ax, b, 10)
    print(np.linalg.eigvalsh(h1))

    print("lanczos on ill-conditioned problem with noise")

    def f_ax_noisy(x):
        return a.dot(x) + np.random.randn(x.size) * 1e-3

    q, h1 = lanczos2(f_ax_noisy, b, 10)
    print(np.linalg.eigvalsh(h1))


if __name__ == "__main__":
    test_lanczos()
    test_cg()
