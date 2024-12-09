
# 📖 Recycly API Documentation

This API allows you to verify waste collection by predicting the classification of bottles (damaged or intact) based on images provided by the user. The predictions are made using a pre-trained machine learning model hosted on Google Cloud Storage.

## 📋 Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Verify Waste Collection](#verify-waste-collection)
- [Request Example](#request-example)


## Base URL

The base URL for this API is:

https://ml-prediction-api-412072547738.asia-southeast2.run.app/verifyWasteCollection


### Authentication

This API does not require authentication for use. However, it requires valid Google Cloud credentials to access the model from Google Cloud Storage. To authenticate:

1. Set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to the path of your Google Cloud Service Account credentials JSON file.
2. Ensure the service account has the required permissions for accessing Google Cloud Storage (e.g., `Storage Object Viewer` or `Storage Admin`).

---

## Endpoints

### Verify Waste Collection

`POST /verifyWasteCollection`

This endpoint accepts an image of a bottle and predicts whether it is intact or damaged.

#### Request

**Headers:**
- `Content-Type`: `multipart/form-data`

**Body:**
- `image`: The image file to be classified. Supported formats include JPEG and PNG.

#### Response

The server responds with a JSON object containing the predicted class, confidence score, and a point value.

**Success Response (200 OK):**

```json
{
    "confidence": 0.9999983310699463,
    "label": "Botol Utuh 1",
    "points": 2,
    "status": "botol diterima"
}
```

```json
{
    "confidence": 0.9979183077812195,
    "label": "Botol Utuh 2",
    "points": 1,
    "status": "botol diterima"
}
```

```json
{
    "confidence": 0.9733534455299377,
    "label": "Bukan Botol",
    "points": 0,
    "status": "botol ditolak"
}
```

```json
{
    "confidence": 0.9999915361404419,
    "label": "Botol Rusak",
    "points": 0,
    "status": "botol ditolak"
}
```

**Error Response (400 Bad Request):**

```json
{
  "error": "No image uploaded"
}
```

**Error: Internal Server Error (500)**
```json
{
  "error": "Internal server error"
}
```

## Request Example
### Request with curl:

```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:8080/verifyWasteCollection
```
### Request with Postman:
    1. Set the method to POST.
    2. Set the URL to http://localhost:8080/verifyWasteCollection.
    3. In the Body section, select form-data and upload the image file under the key image.
### Penjelasan Struktur Dokumentasi:

- **Base URL**: Menjelaskan URL dasar API.
- **Authentication**: Memberikan instruksi tentang cara mengonfigurasi kredensial untuk mengakses Google Cloud Storage.
- **Endpoints**: Menjelaskan endpoint `POST /verifyWasteCollection`, termasuk format request dan response.
- **Request Example**: Memberikan contoh cara mengirim request menggunakan `curl` atau Postman.
- **Response Example**: Menyediakan contoh respons baik untuk kasus sukses maupun error.

