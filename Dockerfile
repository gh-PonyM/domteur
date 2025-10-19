ARG PYTHON_VERSION=3.14
ARG IMAGE_FLAVOR=slim-bookworm
FROM python:$PYTHON_VERSION-$IMAGE_FLAVOR AS python-base
ENV APP_HOME=/app

# System Dependencies
RUN mkdir -p $APP_HOME \
    && useradd -u 1000 -r dev --create-home \
    # all project specific folders need to be accessible by newly created user but also for unknown users (when UID is set manually). Such users are in group root.
    && chown -R dev:root /home/dev \
    && chmod -R 770 /home/dev \
    && apt-get update && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
    git \
    && git config --system init.defaultBranch main \
    # https://github.blog/2022-04-18-highlights-from-git-2-36/#stricter-repository-ownership-checks
    && git config --global safe.directory '*'

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
RUN pip install uv

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
USER dev
