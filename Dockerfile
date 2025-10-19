ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION-bullseye as python-base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

# System Dependencies
RUN mkdir -p /app \
    && useradd -u 1000 -r dev --create-home \
    # all project specific folders need to be accessible by newly created user but also for unknown users (when UID is set manually). Such users are in group root.
    && chown -R dev:root /home/dev \
    && chmod -R 770 /home/dev \
    && apt-get update && apt-get upgrade -y

FROM python-base as development-build

# needs to be set for users with manually set UID
ENV HOME=/home/dev

# important for project bootstrap
WORKDIR $APP_HOME
RUN pip install -U uv

FROM development-build as development

COPY pyproject.toml uv.lock $APP_HOME/

RUN uv sync --no-dev
COPY . $APP_HOME
RUN uv sync
CMD ['uv', 'run', 'domteur', '--version']
