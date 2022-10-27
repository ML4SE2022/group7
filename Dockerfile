FROM python:3.7

ARG TASK_ARGS=""
ARG TASK_URL=https://github.com/go-task/task/releases/download/v3.15.2/task_linux_amd64.deb

RUN curl -o task.deb -L ${TASK_URL} \
    && dpkg -i task.deb \
    && rm task.deb

WORKDIR /project

# Install dependencies
COPY requirements.txt Taskfile.yml ./
RUN --mount=type=cache,target=/root/.cache/pip,mode=0777 task venv:create

COPY CoSQA/ CoSQA/
COPY data/ data/

# Download and unpack model
RUN --mount=type=cache,target=/project/data/model_download,mode=0777 task ${TASK_ARGS} data:unpack-model

RUN task data:preprocess

COPY evaluator/ evaluator/
COPY code/ code/

ENTRYPOINT ["/bin/bash", "-c"]
