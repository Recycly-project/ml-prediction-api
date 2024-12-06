# Menggunakan image Python
FROM python:3.9-slim

# Install git dan git-lfs (jika menggunakan Git LFS untuk model)
RUN apt-get update && apt-get install -y git git-lfs
RUN git lfs install

# Set working directory
WORKDIR /src/app

# Menyalin semua file ke dalam container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Debug untuk memeriksa folder model sebelum menyalin file
RUN echo "Contents before COPY:" && ls -lh

# Menyalin folder model ke dalam folder /model di dalam container
COPY model /model

# Debug untuk memeriksa isi folder model setelah menyalin file
RUN echo "Contents after COPY:" && ls -lh /model

# Expose port 8080 untuk aplikasi
EXPOSE 8080

# Menjalankan aplikasi menggunakan gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.main:app"]
