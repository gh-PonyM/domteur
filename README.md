# domteur

## Installation

Install the command line tool using git repo:
    
    # Using pipx
    pipx install --user git+ssh://git@gitlab.com/pony_m/domteur.git#main
    pipx install git+https://gitlab.com/pony_m/domteur.git#main

    # User install on system
    pip install --user git+https://gitlab.com/pony_m/domteur.git#main

    domteur --version

## Docker

Build the dev image:

     docker build --target development -t domteur:dev .

And run the image:

    docker run -it --rm domteur:dev domteur --help

## Features

- Todo