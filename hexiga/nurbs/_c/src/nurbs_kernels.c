/*
 * nurbs_kernels.c -- original C implementation of a handful of
 * performance-critical B-spline/NURBS routines, exposed as a flat C ABI
 * for use from Python via ctypes.
 *
 * These implement the well-known, textbook ("The NURBS Book", Piegl &
 * Tiller) algorithms for: knot-span location, basis-function evaluation,
 * curve evaluation, derivative control points, knot insertion and degree
 * elevation. The algorithm *descriptions* are standard public-domain
 * numerical-methods material; this file is an independent, from-scratch
 * implementation (own variable naming/structuring, no third-party source
 * code was copied) written specifically for this project.
 *
 * All arrays are plain row-major `double*` buffers; the caller (Python,
 * via hexiga/nurbs/_c/__init__.py) is responsible for allocation and for
 * matching the exact shapes documented per function.  This keeps the ABI
 * minimal and dependency-free (no MATLAB/mex, no external NURBS library).
 */

#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#if defined(_WIN32)
  #define HEXIGA_EXPORT __declspec(dllexport)
#else
  #define HEXIGA_EXPORT __attribute__((visibility("default")))
#endif

#if defined(_OPENMP)
  #include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* findspan -- locate the knot span index containing parameter u.     */
/*                                                                    */
/*   n_ctrl : number of control points                                */
/*   degree : spline degree p                                        */
/*   u      : parameter value                                         */
/*   knots  : knot vector, length n_ctrl + degree + 1                 */
/*   returns: span index i such that knots[i] <= u < knots[i+1]       */
/*            (clamped so the last span is returned at the right end) */
/* ------------------------------------------------------------------ */
HEXIGA_EXPORT
int hexiga_findspan(int n_ctrl, int degree, double u, const double *knots)
{
    int n = n_ctrl - 1;

    if (u >= knots[n + 1]) {
        return n;
    }
    if (u <= knots[degree]) {
        return degree;
    }

    int lo = degree;
    int hi = n + 1;
    int mid = (lo + hi) / 2;
    while (u < knots[mid] || u >= knots[mid + 1]) {
        if (u < knots[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    return mid;
}

/* ------------------------------------------------------------------ */
/* basisfun -- the (degree+1) non-zero B-spline basis functions at u, */
/* active on knot span `span` (as returned by hexiga_findspan).       */
/*                                                                    */
/*   out : pre-allocated buffer of length degree + 1                  */
/* ------------------------------------------------------------------ */
HEXIGA_EXPORT
void hexiga_basisfun(int span, double u, int degree, const double *knots,
                      double *out)
{
    double left[64];
    double right[64];

    out[0] = 1.0;
    for (int j = 1; j <= degree; ++j) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            double denom = right[r + 1] + left[j - r];
            double temp = (denom != 0.0) ? out[r] / denom : 0.0;
            out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        out[j] = saved;
    }
}

/* ------------------------------------------------------------------ */
/* bspeval -- evaluate a B-spline curve (row-major control points) at  */
/* a batch of parameter values.                                       */
/*                                                                    */
/*   c        : (n_rows, n_ctrl) row-major control-point array        */
/*   knots    : length n_ctrl + degree + 1                            */
/*   u        : length n_u, parameter samples                         */
/*   out      : pre-allocated (n_rows, n_u) row-major output          */
/* ------------------------------------------------------------------ */
HEXIGA_EXPORT
void hexiga_bspeval(int degree, const double *c, int n_rows, int n_ctrl,
                     const double *knots, const double *u, int n_u,
                     double *out)
{
    /* Each column (parameter sample) is fully independent, so this loop
     * parallelises trivially across threads when built with OpenMP; the
     * per-column basis-function buffer `N` is declared inside the loop
     * body so it is private to each iteration/thread. */
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (n_u > 64)
    #endif
    for (int col = 0; col < n_u; ++col) {
        double N[64];
        int span = hexiga_findspan(n_ctrl, degree, u[col], knots);
        hexiga_basisfun(span, u[col], degree, knots, N);
        int first = span - degree;
        for (int row = 0; row < n_rows; ++row) {
            double acc = 0.0;
            const double *crow = c + (size_t)row * n_ctrl;
            for (int ii = 0; ii <= degree; ++ii) {
                acc += N[ii] * crow[first + ii];
            }
            out[(size_t)row * n_u + col] = acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/* bspderiv -- control points / knot vector of the first derivative   */
/* of a B-spline curve.                                                */
/*                                                                    */
/*   c   : (n_rows, n_ctrl) row-major                                  */
/*   knots : length n_ctrl + degree + 1                               */
/*   dc  : pre-allocated (n_rows, n_ctrl - 1) row-major output         */
/*   dk  : pre-allocated (n_ctrl + degree - 1,) output                 */
/*         (== knots[1 : n_ctrl + degree], i.e. interior knots)        */
/* ------------------------------------------------------------------ */
HEXIGA_EXPORT
void hexiga_bspderiv(int degree, const double *c, int n_rows, int n_ctrl,
                      const double *knots, double *dc, double *dk)
{
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (n_ctrl > 256)
    #endif
    for (int i = 0; i < n_ctrl - 1; ++i) {
        double span = knots[i + degree + 1] - knots[i + 1];
        double scale = (span != 0.0) ? (double)degree / span : 0.0;
        for (int row = 0; row < n_rows; ++row) {
            const double *crow = c + (size_t)row * n_ctrl;
            dc[(size_t)row * (n_ctrl - 1) + i] = scale * (crow[i + 1] - crow[i]);
        }
    }
    int n_dk = n_ctrl + degree - 1;
    for (int i = 0; i < n_dk; ++i) {
        dk[i] = knots[i + 1];
    }
}

/* ------------------------------------------------------------------ */
/* bspkntins -- insert a (non-decreasing) batch of knot values u into  */
/* a B-spline, producing refined control points/knots.                */
/*                                                                    */
/*   c   : (n_rows, n_ctrl) row-major                                  */
/*   knots : length n_ctrl + degree + 1                                */
/*   u   : length n_u, non-decreasing parameter values to insert       */
/*   ic  : pre-allocated (n_rows, n_ctrl + n_u) row-major output        */
/*   ik  : pre-allocated (n_ctrl + degree + 1 + n_u,) output            */
/* ------------------------------------------------------------------ */
HEXIGA_EXPORT
void hexiga_bspkntins(int degree, const double *c, int n_rows, int n_ctrl,
                       const double *knots, const double *u, int n_u,
                       double *ic, double *ik)
{
    int n = n_ctrl - 1;
    int r = n_u - 1;
    int m = n + degree + 1;
    int n_ic_cols = n_ctrl + n_u;

    int a = hexiga_findspan(n_ctrl, degree, u[0], knots);
    int b = hexiga_findspan(n_ctrl, degree, u[r], knots) + 1;

#define IC(row, col) ic[(size_t)(row) * n_ic_cols + (col)]
#define C(row, col) c[(size_t)(row) * n_ctrl + (col)]

    for (int q = 0; q < n_rows; ++q) {
        for (int j = 0; j <= a - degree; ++j) {
            IC(q, j) = C(q, j);
        }
        for (int j = b - 1; j <= n; ++j) {
            IC(q, j + r + 1) = C(q, j);
        }
    }

    for (int j = 0; j <= a; ++j) {
        ik[j] = knots[j];
    }
    for (int j = b + degree; j <= m; ++j) {
        ik[j + r + 1] = knots[j];
    }

    int i = b + degree - 1;
    int s = b + degree + r;
    for (int j = r; j >= 0; --j) {
        while (u[j] <= knots[i] && i > a) {
            for (int q = 0; q < n_rows; ++q) {
                IC(q, s - degree - 1) = C(q, i - degree - 1);
            }
            ik[s] = knots[i];
            --s;
            --i;
        }
        for (int q = 0; q < n_rows; ++q) {
            IC(q, s - degree - 1) = IC(q, s - degree);
        }
        for (int l = 1; l <= degree; ++l) {
            int ind = s - degree + l;
            double alfa = ik[s + l] - u[j];
            if (fabs(alfa) == 0.0) {
                for (int q = 0; q < n_rows; ++q) {
                    IC(q, ind - 1) = IC(q, ind);
                }
            } else {
                alfa = alfa / (ik[s + l] - knots[i - degree + l]);
                for (int q = 0; q < n_rows; ++q) {
                    IC(q, ind - 1) = alfa * IC(q, ind - 1) + (1.0 - alfa) * IC(q, ind);
                }
            }
        }
        ik[s] = u[j];
        --s;
    }

#undef IC
#undef C
}

/* ------------------------------------------------------------------ */
/* bspdegelev -- elevate a B-spline curve from `degree` to             */
/* `degree + raise_by`.                                                */
/*                                                                    */
/* Output size is data-dependent (it depends on interior-knot          */
/* multiplicities), so the caller must over-allocate `ic`/`ik` to a    */
/* safe upper bound (e.g. n_ctrl * (raise_by + 1) columns and knots)   */
/* and this function writes the true output size into *n_ic_out /     */
/* *n_ik_out.                                                          */
/* ------------------------------------------------------------------ */
static double binomial(int n, int k)
{
    if (k < 0 || k > n) return 0.0;
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result *= (double)(n - i) / (double)(i + 1);
    }
    return result;
}

HEXIGA_EXPORT
void hexiga_bspdegelev(int degree, const double *c, int n_rows, int n_ctrl,
                        const double *knots, int raise_by,
                        double *ic, int ic_stride,
                        double *ik, int *n_ic_out, int *n_ik_out)
{
    int n = n_ctrl - 1;
    int m = n + degree + 1;
    int ph = degree + raise_by;
    int ph2 = ph / 2;

    /* bezalfs[i * (degree+1) + j], i in 0..ph, j in 0..degree */
    int d1 = degree + 1;
    double *bezalfs = (double *)calloc((size_t)(ph + 1) * d1, sizeof(double));
    bezalfs[0 * d1 + 0] = 1.0;
    bezalfs[ph * d1 + degree] = 1.0;

    for (int i = 1; i <= ph2; ++i) {
        double inv = 1.0 / binomial(ph, i);
        int mpi = (degree < i) ? degree : i;
        for (int j = (i - raise_by > 0 ? i - raise_by : 0); j <= mpi; ++j) {
            bezalfs[i * d1 + j] = inv * binomial(degree, j) * binomial(raise_by, i - j);
        }
    }
    for (int i = ph2 + 1; i < ph; ++i) {
        int mpi = (degree < i) ? degree : i;
        for (int j = (i - raise_by > 0 ? i - raise_by : 0); j <= mpi; ++j) {
            bezalfs[i * d1 + j] = bezalfs[(ph - i) * d1 + (degree - j)];
        }
    }

    double *bpts = (double *)malloc((size_t)d1 * n_rows * sizeof(double));
    double *ebpts = (double *)malloc((size_t)(ph + 1) * n_rows * sizeof(double));
    double *next_bpts = (double *)malloc((size_t)d1 * n_rows * sizeof(double));
    double *alfs = (double *)malloc((size_t)d1 * sizeof(double));

#define C(row, col) c[(size_t)(row) * n_ctrl + (col)]
#define BPTS(i, row) bpts[(size_t)(i) * n_rows + (row)]
#define EBPTS(i, row) ebpts[(size_t)(i) * n_rows + (row)]
#define NEXTBPTS(i, row) next_bpts[(size_t)(i) * n_rows + (row)]
#define IC(row, col) ic[(size_t)(row) * ic_stride + (col)]

    int mh = ph;
    int kind = ph + 1;
    int r = -1;
    int a = degree;
    int b = degree + 1;
    int cind = 1;
    double ua = knots[0];

    for (int row = 0; row < n_rows; ++row) {
        IC(row, 0) = C(row, 0);
    }
    for (int i = 0; i <= ph; ++i) {
        ik[i] = ua;
    }

    for (int i = 0; i <= degree; ++i) {
        for (int row = 0; row < n_rows; ++row) {
            BPTS(i, row) = C(row, i);
        }
    }

    while (b < m) {
        int i = b;
        while (b < m && knots[b] == knots[b + 1]) {
            ++b;
        }
        int mul = b - i + 1;
        mh += mul + raise_by;
        double ub = knots[b];
        int oldr = r;
        r = degree - mul;

        int lbz = (oldr > 0) ? (oldr + 2) / 2 : 1;
        int rbz = (r > 0) ? ph - (r + 1) / 2 : ph;

        if (r > 0) {
            double numer = ub - ua;
            for (int q = degree; q > mul; --q) {
                alfs[q - mul - 1] = numer / (knots[a + q] - ua);
            }
            for (int j = 1; j <= r; ++j) {
                int save = r - j;
                int s = mul + j;
                for (int q = degree; q >= s; --q) {
                    for (int row = 0; row < n_rows; ++row) {
                        BPTS(q, row) = alfs[q - s] * BPTS(q, row)
                                     + (1.0 - alfs[q - s]) * BPTS(q - 1, row);
                    }
                }
                for (int row = 0; row < n_rows; ++row) {
                    NEXTBPTS(save, row) = BPTS(degree, row);
                }
            }
        }

        for (int ii = lbz; ii <= ph; ++ii) {
            for (int row = 0; row < n_rows; ++row) {
                EBPTS(ii, row) = 0.0;
            }
            int mpi = (degree < ii) ? degree : ii;
            for (int j = (ii - raise_by > 0 ? ii - raise_by : 0); j <= mpi; ++j) {
                double alfa = bezalfs[ii * d1 + j];
                for (int row = 0; row < n_rows; ++row) {
                    EBPTS(ii, row) += alfa * BPTS(j, row);
                }
            }
        }

        if (oldr > 1) {
            int first = kind - 2;
            int last = kind;
            double den = ub - ua;
            double bet = (ik[kind - 1] != ua) ? (ub - ik[kind - 1]) / den : 0.0;

            for (int tr = 1; tr < oldr; ++tr) {
                int ii = first;
                int jj = last;
                int kj = jj - kind + 1;
                while (jj - ii > tr) {
                    if (ii < cind) {
                        double alf = (ua - ik[ii] != 0.0) ? (ub - ik[ii]) / (ua - ik[ii]) : 0.0;
                        for (int row = 0; row < n_rows; ++row) {
                            IC(row, ii) = alf * IC(row, ii) + (1.0 - alf) * IC(row, ii - 1);
                        }
                    }
                    if (jj >= lbz) {
                        if (jj - tr <= kind - ph + oldr) {
                            double gam = (ub - ik[jj - tr]) / den;
                            for (int row = 0; row < n_rows; ++row) {
                                EBPTS(kj, row) = gam * EBPTS(kj, row) + (1.0 - gam) * EBPTS(kj + 1, row);
                            }
                        } else {
                            for (int row = 0; row < n_rows; ++row) {
                                EBPTS(kj, row) = bet * EBPTS(kj, row) + (1.0 - bet) * EBPTS(kj + 1, row);
                            }
                        }
                    }
                    ++ii;
                    --jj;
                    --kj;
                }
                --first;
                ++last;
            }
        }

        if (a != degree) {
            for (int cnt = 0; cnt < ph - oldr; ++cnt) {
                ik[kind] = ua;
                ++kind;
            }
        }

        for (int j = lbz; j <= rbz; ++j) {
            for (int row = 0; row < n_rows; ++row) {
                IC(row, cind) = EBPTS(j, row);
            }
            ++cind;
        }

        if (b < m) {
            for (int j = 0; j < r; ++j) {
                for (int row = 0; row < n_rows; ++row) {
                    BPTS(j, row) = NEXTBPTS(j, row);
                }
            }
            for (int j = r; j <= degree; ++j) {
                for (int row = 0; row < n_rows; ++row) {
                    BPTS(j, row) = C(row, b - degree + j);
                }
            }
            a = b;
            ++b;
            ua = ub;
        } else {
            for (int cnt = 0; cnt <= ph; ++cnt) {
                ik[kind + cnt] = ub;
            }
        }
    }

    int n_ic = mh - ph;
    int n_ik = n_ic + degree + raise_by + 1;
    *n_ic_out = n_ic;
    *n_ik_out = n_ik;

#undef IC
#undef NEXTBPTS
#undef EBPTS
#undef BPTS
#undef C

    free(bezalfs);
    free(bpts);
    free(ebpts);
    free(next_bpts);
    free(alfs);
}
