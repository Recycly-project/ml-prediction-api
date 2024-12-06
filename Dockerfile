FROM python:3.9-slim

WORKDIR /src/app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Debugging: Tampilkan isi folder model
RUN echo "Verifying model file in Docker build:" && ls -lh /model

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.main:app"]
