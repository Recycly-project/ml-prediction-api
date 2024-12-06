FROM python:3.9-slim

WORKDIR /src/app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Debug untuk memeriksa folder model sebelum copy
RUN echo "Contents before COPY:" && ls -lh
COPY model /model
RUN echo "Contents after COPY:" && ls -lh /model
EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.main:app"]
