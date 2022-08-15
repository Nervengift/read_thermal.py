FROM python:3.9

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH="${PATH}:/root/.poetry/bin"

RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt install -y \
    build-essential \
    curl \
    exiftool \
    git

# Allow root for Jupyter notebooks
RUN mkdir /root/.jupyter
RUN echo "c.NotebookApp.allow_root = True" > /root/.jupyter/jupyter_notebook_config.py

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false
  
COPY pyproject.toml poetry.lock* ./

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=true
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

CMD mkdir -p /code
WORKDIR /code

ADD . .
ENTRYPOINT [ "make" ]