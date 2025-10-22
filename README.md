# domteur

## Installation

Install the command line tool using git repo:
    
    # Using pipx
    pipx install --user git+ssh://git@github.com/gh-PonyM/domteur.git#main
    pipx install git+https://github.com/gh-PonyM/domteur.git#main

    # User install on system
    pip install --user git+https://github.com/gh-PonyM/domteur.git#main

    domteur --version

## Starting components

Start the broker first:

    docker compose up broker

### LLM

     uv run domteur --cfg-file config.example.yml llm start

### Terminal

    uv run domteur --cfg-file config.example.yml chat repl


## Docker

Build the dev image:

     docker compose build llm

And run the component with the broker:

    docker compose up

## Features

- Todo