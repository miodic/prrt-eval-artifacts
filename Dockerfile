FROM debian:13.4

ENV DEBIAN_FRONTEND=noninteractive

# Install core system dependencies (added ca-certificates for secure curl/uv downloads)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    iproute2 \ 
    libcap2-bin \
    sudo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install specific Rust version
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add both Cargo and uv to the PATH
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Set working directory for setup
WORKDIR /artifact

# Copy requirements.txt first to leverage Docker cache
# (Note: if your requirements.txt is actually inside the python/ folder, change this to COPY python/requirements.txt .)
COPY python/requirements.txt .

# Use uv to install Python 3.13 and set up the virtual environment
RUN uv python install 3.13 && \
    uv venv /opt/venv --python 3.13

# Activate the venv for subsequent commands
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"

# Install dependencies using uv pip
RUN uv pip install -r requirements.txt

# Copy the entire repository into the container
# WARNING: make sure you ran `git submodule update --init` before!
COPY . .

# Build the root Rust project and set capabilities
RUN cargo build --bin main --release
RUN setcap cap_net_admin,cap_sys_nice=eip ./target/release/main

# Build the inc_search_eval Rust project and set capabilities
WORKDIR /artifact/inc_search_eval
RUN cargo build --release
RUN setcap cap_net_admin,cap_sys_nice=eip ./target/release/inc_search_eval

# Switch to the Python directory for Jupyter
WORKDIR /artifact/python

EXPOSE 8888

# Start Jupyter Lab using uv run, without token/password authentication
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]