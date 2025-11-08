ARG BASE_IMAGE=python:3.12-slim
ARG APP_USER=ubuntu
FROM $BASE_IMAGE AS python-base
ENV APP_HOME=/app \
  DOMTEUR_CONFIG_DIR=/app

# System Dependencies
# RUN useradd -u 1000 -r dev --create-home
RUN if ! id -u 1000 >/dev/null 2>&1; then \
      useradd -u 1000 -r dev --create-home; \
    fi
# all project specific folders need to be accessible by newly created user but also for unknown users (when UID is set manually). Such users are in group root
RUN mkdir -p $APP_HOME && chown -R $APP_USER:root /home/$APP_USER && chmod -R 770 /home/$APP_USER
RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  gcc \
  git \
  portaudio19-dev \
  alsa-utils \
  pulseaudio
RUN git config --system init.defaultBranch main
# https://github.blog/2022-04-18-highlights-from-git-2-36/#stricter-repository-ownership-check
RUN git config --global safe.directory '*'
RUN rm -rf /var/lib/apt/lists/

ENV PATH="/app/.venv/bin:$PATH"

FROM python-base AS development-build
# needs to be set for users with manually set UID
ENV HOME=/home/dev \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

# important for project bootstrap in order to create the uv.lock file
RUN pip install uv --break-system-packages

FROM development-build AS development

WORKDIR $APP_HOME
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY --chown=dev:dev pyproject.toml uv.lock $APP_HOME/
COPY --chown=dev:dev tests $APP_HOME/tests
COPY --chown=dev:dev domteur $APP_HOME/domteur
COPY --chown=dev:dev LICENSE $APP_HOME/LICENSE

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python-base AS production

WORKDIR $APP_HOME
COPY --from=development --chown=dev:dev $APP_HOME $APP_HOME
USER $APP_USER
