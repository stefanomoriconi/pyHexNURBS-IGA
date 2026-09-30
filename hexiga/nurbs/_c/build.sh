#!/usr/bin/env bash
# Build the optional native NURBS-kernel accelerator as a shared library.
#
# Usage: bash hexiga/nurbs/_c/build.sh
# Produces hexiga/nurbs/_c/libhexiga_nurbs_c.so (Linux/macOS: .dylib).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$here/src/nurbs_kernels.c"

case "$(uname -s)" in
    Darwin*) out="$here/libhexiga_nurbs_c.dylib" ;;
    *)       out="$here/libhexiga_nurbs_c.so" ;;
esac

# OpenMP is optional (parallelises the per-sample evaluation loop); fall
# back to a sequential build if the toolchain/libomp is unavailable.
if cc -O3 -fPIC -shared -fopenmp -Wall -Wextra -o "$out" "$src" -lm 2>/tmp/hexiga_c_build.log; then
    echo "Built $out (OpenMP enabled)"
else
    echo "OpenMP build failed, falling back to sequential build:" >&2
    cat /tmp/hexiga_c_build.log >&2
    cc -O3 -fPIC -shared -Wall -Wextra -o "$out" "$src" -lm
    echo "Built $out (sequential)"
fi
