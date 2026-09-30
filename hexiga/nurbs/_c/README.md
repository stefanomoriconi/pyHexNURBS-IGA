# `hexiga.nurbs._c` — optional C accelerator

This is an **optional**, original C implementation of the six core NURBS
kernels (`findspan`, `basisfun`, `bspeval`, `bspderiv`, `bspkntins`,
`bspdegelev`), built from scratch from the standard "The NURBS Book"
(Piegl & Tiller) algorithms. It is not required to use the package: the
pure-NumPy implementation in `hexiga.nurbs._kernels` is the default and
remains the deterministic reference used by the test suite.

It exists purely as a speed-up for batched/high-volume evaluation (e.g.
dense sampling for visualization or large assemblies), parallelized with
OpenMP where beneficial.

## Build

```bash
bash hexiga/nurbs/_c/build.sh
```

This compiles `src/nurbs_kernels.c` into a shared library
(`libhexiga_nurbs_c.so` / `.dylib`) next to this README, using `gcc` or
`clang`. OpenMP is used if available, with an automatic sequential
fallback otherwise. No third-party dependencies are required — only a
standard C compiler.

The compiled library is not committed to the repository; build it locally
if you want the accelerated path.

## Activate

```bash
export HEXIGA_BACKEND=nurbs_c
```

With this environment variable set (and the shared library built), all
NURBS operations transparently use the C kernels instead of NumPy — no
code changes are needed. If the library is missing or fails to import,
the package silently falls back to the NumPy implementation.
