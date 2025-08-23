FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VER="3.10"
ARG USERNAME
ARG UID
ARG GID=${UID}

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    sox libsndfile1 ffmpeg cmake \
    python${PYTHON_VER}-dev \
    python3-pip \
    sudo \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VER} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 1

# Create non-root user
RUN groupadd --gid ${GID} $USERNAME \
    && useradd --uid ${UID} --gid ${GID} -m $USERNAME \
    && usermod -aG sudo $USERNAME \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install Poetry globally
RUN pip install --no-cache-dir poetry

USER ${USERNAME}
WORKDIR /home/${USERNAME}/SchizoNegSympAI

COPY --chown=${USERNAME}:${USERNAME} pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

# Set PATH in .bashrc for future shells
RUN POETRY_BIN=$(poetry env info -p)/bin && \
    echo "export PATH=\"$POETRY_BIN:\$PATH\"" >> ~/.bashrc

CMD ["bash"]
