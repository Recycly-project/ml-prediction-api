FROM python:3.9-slim

RUN apt-get update && apt-get install -y git git-lfs
RUN git lfs install

WORKDIR /src/app

COPY . .
RUN git lfs pull

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN echo "Verifying model file presence in Docker build:" && ls -lh /model

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.main:app"]
