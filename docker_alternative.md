# البديل: استخدام Docker على RunPod

إذا كنت تفضل استخدام Docker، إليك الطريقة:

## 1. إنشاء Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# تثبيت متطلبات النظام
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملفات المشروع
COPY requirements.txt .
COPY . .

# تثبيت Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# تحميل النماذج المدربة
RUN mkdir -p RAFT/models && \
    cd RAFT/models && \
    wget https://github.com/princeton-vl/RAFT/releases/download/models/raft-things.pth && \
    wget https://github.com/princeton-vl/RAFT/releases/download/models/raft-sintel.pth

# تعيين متغيرات البيئة
ENV PYTHONPATH=/app
ENV FLASK_APP=ui/app.py

EXPOSE 5000

CMD ["python", "ui/app.py"]
```

## 2. بناء الصورة
```bash
docker build -t motion-vectorization .
```

## 3. تشغيل الكونتينر
```bash
docker run -d \
  --name motion-app \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/videos:/app/videos \
  motion-vectorization
```

## 4. استخدام Docker Compose

```yaml
version: '3.8'
services:
  motion-vectorization:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./videos:/app/videos
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

شغّل بـ:
```bash
docker-compose up -d
```