# FROM python:3.9-slim

# WORKDIR /src/app

# COPY . .

# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt

# EXPOSE 8080

# CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.main:app"]
