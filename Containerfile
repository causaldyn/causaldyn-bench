# The bench's reproduction image: CPU only, every package pinned by the bench's locks.
#
# Build it from a directory that holds this repository and the library side by side, as uv.lock's
# path dependency reads them, the library at the tag of the version uv.lock names:
#
#   git clone https://github.com/causaldyn/causaldyn-bench
#   git clone --branch v0.23.0 https://github.com/causaldyn/causal-hybrid-control
#   podman build -f causaldyn-bench/Containerfile -t causaldyn-bench-reproduce .
#   podman run --rm causaldyn-bench-reproduce            # scripts/reproduce.sh
#
# What it holds:
# * the bench's environment, uv.lock's, frozen, on CPython 3.14.8, the interpreter the committed
#   cells (results/scorecard_reproduce.json) were recorded on;
# * the tools' arms' environments under /opt/rivals, which CAUSALDYN_RIVALS names and
#   scripts/reproduce.sh finds: PyMC-Marketing's, Meridian's and Orbit's on Python 3.12, each from
#   its pinned file scripts/<arm>.txt, Orbit's torch the CPU build from PyTorch's own index;
#   Robyn's R library from scripts/robyn_arm.lock.txt, every package at its locked version from
#   CRAN's sources or its archive, built with the lock's flags, and nevergrad's environment on
#   Fedora's Python 3.13 from the same file's pins.
#
# Fedora 44 is the base because the Robyn arm's lock was built and run on it, R and Python 3.13
# from its packages; R itself is the release's current build, which the build prints. Robyn's
# library compiles rstan, StanHeaders and prophet from source: the build takes hours.
FROM registry.fedoraproject.org/fedora:44

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

RUN dnf -y install --setopt=install_weak_deps=False \
        R-core R-core-devel gcc gcc-c++ gcc-gfortran make cmake curl git which python3.13 \
        libcurl-devel openssl-devel libxml2-devel libicu-devel zlib-devel libpng-devel NLopt-devel \
    && dnf clean all \
    && R --version | head -1

ENV UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    CAUSALDYN_RIVALS=/opt/rivals

# The pins alone first, so that a change to the bench's code rebuilds none of the tools.
COPY causaldyn-bench/scripts/pymc_marketing_arm.txt causaldyn-bench/scripts/meridian_arm.txt \
     causaldyn-bench/scripts/orbit_arm.txt causaldyn-bench/scripts/robyn_arm.lock.txt /pins/

RUN uv venv --python 3.12 /opt/rivals/pymc-marketing \
    && uv pip install --python /opt/rivals/pymc-marketing/bin/python -r /pins/pymc_marketing_arm.txt \
    && uv venv --python 3.12 /opt/rivals/meridian \
    && uv pip install --python /opt/rivals/meridian/bin/python -r /pins/meridian_arm.txt \
    && uv venv --python 3.12 /opt/rivals/orbit \
    && uv pip install --python /opt/rivals/orbit/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        -r /pins/orbit_arm.txt

# nevergrad's environment, from the lock's pins, on the Python 3.13 reticulate embeds
RUN grep -E '^[A-Za-z0-9._-]+==' /pins/robyn_arm.lock.txt > /pins/nevergrad.txt \
    && uv venv --python /usr/bin/python3.13 /opt/rivals/robyn/ngvenv \
    && uv pip install --python /opt/rivals/robyn/ngvenv/bin/python -r /pins/nevergrad.txt

# Robyn's R library: every package the lock names, and the recommended packages it found updated
# in the site library, at its version, gathered into a local repository so that R installs them in
# the order their dependencies need; the build stops where a version is not the lock's.
RUN set -eu; \
    mkdir -p /opt/cran/src/contrib /opt/rivals/robyn/rlib; \
    grep -E '^#   (C|CXX|CXX17|CXX20|F|FC)FLAGS = ' /pins/robyn_arm.lock.txt \
        | sed 's/^#   //' > /opt/cran/Makevars; \
    { grep '^R ' /pins/robyn_arm.lock.txt | awk '{print $2, $3}'; \
      grep '^system .* recommended /usr/local/' /pins/robyn_arm.lock.txt | awk '{print $2, $3}'; \
    } > /opt/cran/wanted.txt; \
    while read -r name version; do \
        file="${name}_${version}.tar.gz"; \
        curl -fsSL -o "/opt/cran/src/contrib/$file" "https://cloud.r-project.org/src/contrib/$file" \
        || curl -fsSL -o "/opt/cran/src/contrib/$file" \
            "https://cloud.r-project.org/src/contrib/Archive/$name/$file"; \
    done < /opt/cran/wanted.txt; \
    Rscript --vanilla -e 'tools::write_PACKAGES("/opt/cran/src/contrib", type = "source")'; \
    R_LIBS=/opt/rivals/robyn/rlib R_LIBS_USER=/opt/rivals/robyn/rlib \
    R_MAKEVARS_USER=/opt/cran/Makevars MAKEFLAGS=-j2 Rscript --vanilla -e ' \
        lib <- "/opt/rivals/robyn/rlib"; \
        wanted <- read.table("/opt/cran/wanted.txt", col.names = c("name", "version"), \
                             colClasses = "character"); \
        install.packages(wanted$name, lib = lib, repos = "file:///opt/cran", type = "source", \
                         dependencies = c("Depends", "Imports", "LinkingTo"), Ncpus = 2); \
        have <- installed.packages(lib.loc = lib)[, "Version"]; \
        off <- wanted[is.na(have[wanted$name]) | have[wanted$name] != wanted$version, ]; \
        if (nrow(off) > 0) { print(off); stop("the R library is not the lock") }'; \
    rm -rf /opt/cran/src

# The bench and the library, side by side, and the bench's environment from uv.lock, frozen
COPY causal-hybrid-control /work/causal-hybrid-control
COPY causaldyn-bench /work/causaldyn-bench
WORKDIR /work/causaldyn-bench
RUN uv python install 3.14.8 \
    && uv sync --frozen --python 3.14.8 --group dev \
    && uv run --no-sync python -c "import causaldyn_bench, chc; print('chc', chc.__version__)"

CMD ["scripts/reproduce.sh"]
