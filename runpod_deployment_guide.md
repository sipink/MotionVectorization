# دليل نشر مشروع Motion Vectorization على RunPod

## الخطوة 1: إعداد RunPod Instance

### 1.1 إنشاء Instance جديد
1. اذهب إلى [RunPod Console](https://www.runpod.io/console/pods)
2. انقر على "Deploy" → "GPU Cloud"
3. اختر Template أو Custom:
   - **الموصى به**: PyTorch 2.0.0 + Python 3.10
   - أو استخدم Ubuntu 22.04 LTS للتحكم الكامل

### 1.2 المواصفات المطلوبة
```
GPU: RTX 3080 أو أعلى (للأداء الأمثل)
CPU: 8 cores+
RAM: 32GB+
Storage: 50GB+ (للنماذج والفيديوهات)
```

### 1.3 تكوين الشبكة والمنافذ
- تأكد من تفعيل SSH (Port 22)
- تفعيل HTTP (Port 5000) للتطبيق
- تفعيل HTTPS (Port 443) إذا كان مطلوباً

## الخطوة 2: الاتصال عبر SSH

```bash
# احصل على IP Address من RunPod Console
ssh root@YOUR_POD_IP

# أو إذا كنت تستخدم مفتاح SSH
ssh -i ~/.ssh/id_rsa root@YOUR_POD_IP
```

## الخطوة 3: إعداد البيئة

### 3.1 تحديث النظام
```bash
apt update && apt upgrade -y
apt install -y git curl wget build-essential
```

### 3.2 تثبيت Python 3.11 (إذا لم يكن متوفراً)
```bash
apt install -y python3.11 python3.11-pip python3.11-venv
ln -sf /usr/bin/python3.11 /usr/bin/python3
ln -sf /usr/bin/pip3.11 /usr/bin/pip3
```

## الخطوة 4: نقل ملفات المشروع

### 4.1 إنشاء مجلد المشروع
```bash
mkdir -p /workspace/motion-vectorization
cd /workspace/motion-vectorization
```

### 4.2 نقل الملفات (من جهازك المحلي)
```bash
# من Terminal محلي (ليس SSH)
scp -r ./RAFT root@YOUR_POD_IP:/workspace/motion-vectorization/
scp -r ./motion_vectorization root@YOUR_POD_IP:/workspace/motion-vectorization/
scp -r ./svg_utils root@YOUR_POD_IP:/workspace/motion-vectorization/
scp -r ./scripts root@YOUR_POD_IP:/workspace/motion-vectorization/
scp -r ./ui root@YOUR_POD_IP:/workspace/motion-vectorization/
scp -r ./videos root@YOUR_POD_IP:/workspace/motion-vectorization/
scp ./pyproject.toml root@YOUR_POD_IP:/workspace/motion-vectorization/
scp ./requirements.txt root@YOUR_POD_IP:/workspace/motion-vectorization/
```

### 4.3 بديل: استخدام Git (إذا كان المشروع على GitHub)
```bash
# إذا كان المشروع محفوظ على Git
git clone YOUR_REPOSITORY_URL /workspace/motion-vectorization
cd /workspace/motion-vectorization
```

## الخطوة 5: تثبيت المتطلبات

### 5.1 إنشاء بيئة افتراضية
```bash
cd /workspace/motion-vectorization
python3 -m venv venv
source venv/bin/activate
```

### 5.2 تثبيت PyTorch مع دعم CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5.3 تثبيت باقي المتطلبات
```bash
pip install -r requirements.txt
```

## الخطوة 6: إعداد التطبيق

### 6.1 تكوين Flask للوصول الخارجي
تحقق من أن `ui/app.py` يحتوي على:
```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
```

### 6.2 تحميل النماذج المدربة (إذا لم تكن موجودة)
```bash
cd RAFT/models
# تحميل النماذج حسب الحاجة
wget https://github.com/princeton-vl/RAFT/releases/download/models/raft-things.pth
wget https://github.com/princeton-vl/RAFT/releases/download/models/raft-sintel.pth
```

## الخطوة 7: إعداد Firewall والأمان

### 7.1 تكوين UFW (إذا كان مُفعّلاً)
```bash
ufw allow 22/tcp    # SSH
ufw allow 5000/tcp  # Flask App
ufw enable
```

### 7.2 إعداد Nginx كـ Reverse Proxy (اختياري)
```bash
apt install -y nginx

# إنشاء تكوين Nginx
cat > /etc/nginx/sites-available/motion-vectorization << 'EOF'
server {
    listen 80;
    server_name YOUR_POD_IP;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# تفعيل التكوين
ln -s /etc/nginx/sites-available/motion-vectorization /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx
```

## الخطوة 8: تشغيل التطبيق

### 8.1 تشغيل مباشر (للاختبار)
```bash
cd /workspace/motion-vectorization
source venv/bin/activate
python ui/app.py
```

### 8.2 تشغيل باستخدام Screen (للعمل في الخلفية)
```bash
screen -S motion-app
cd /workspace/motion-vectorization
source venv/bin/activate
python ui/app.py
# اضغط Ctrl+A ثم D للخروج من Screen

# للعودة إلى Session
screen -r motion-app
```

### 8.3 إنشاء Systemd Service (للتشغيل التلقائي)
```bash
cat > /etc/systemd/system/motion-vectorization.service << 'EOF'
[Unit]
Description=Motion Vectorization Flask App
After=network.target

[Service]
User=root
WorkingDirectory=/workspace/motion-vectorization
Environment=PATH=/workspace/motion-vectorization/venv/bin
ExecStart=/workspace/motion-vectorization/venv/bin/python ui/app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable motion-vectorization
systemctl start motion-vectorization
systemctl status motion-vectorization
```

## الخطوة 9: اختبار التطبيق

### 9.1 اختبار محلي على الخادم
```bash
curl http://localhost:5000
```

### 9.2 اختبار من المتصفح
افتح في المتصفح:
```
http://YOUR_POD_IP:5000
```

## الخطوة 10: مراقبة الأداء

### 10.1 مراقبة الذاكرة والمعالج
```bash
htop
nvidia-smi  # لمراقبة GPU
```

### 10.2 مراقبة سجلات التطبيق
```bash
# إذا كنت تستخدم systemd
journalctl -u motion-vectorization -f

# إذا كنت تستخدم screen
screen -r motion-app
```

## ملاحظات مهمة

1. **الأمان**: تأكد من تغيير كلمات المرور الافتراضية
2. **النسخ الاحتياطي**: احفظ نسخة من النماذج والتكوينات
3. **التكلفة**: راقب استهلاك GPU لتجنب التكاليف الزائدة
4. **الشبكة**: تأكد من أن RunPod يسمح بالاتصالات الخارجية للمنفذ 5000

## استكشاف الأخطاء

### مشكلة في الاتصال
```bash
# تحقق من حالة الخدمة
systemctl status motion-vectorization

# تحقق من المنافذ المفتوحة
netstat -tlnp | grep :5000

# تحقق من سجلات النظام
dmesg | tail
```

### مشكلة في الذاكرة
```bash
# تحقق من استهلاك الذاكرة
free -h
df -h

# تنظيف cache إذا لزم الأمر
sync && echo 3 > /proc/sys/vm/drop_caches
```