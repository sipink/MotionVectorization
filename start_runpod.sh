#!/bin/bash

# سكريبت تشغيل التطبيق على RunPod
# Motion Vectorization Project Startup Script

echo "🚀 بدء تشغيل تطبيق Motion Vectorization على RunPod..."

# التحقق من وجود Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 غير مثبت"
    exit 1
fi

# التحقق من وجود مجلد البيئة الافتراضية
if [ ! -d "venv" ]; then
    echo "📦 إنشاء بيئة افتراضية..."
    python3 -m venv venv
fi

# تفعيل البيئة الافتراضية
echo "🔧 تفعيل البيئة الافتراضية..."
source venv/bin/activate

# تثبيت المتطلبات
echo "📋 تثبيت المتطلبات..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# التحقق من وجود النماذج المدربة
echo "🤖 التحقق من النماذج المدربة..."
if [ ! -f "RAFT/models/raft-things.pth" ]; then
    echo "⬇️  تحميل نماذج RAFT..."
    mkdir -p RAFT/models
    cd RAFT/models
    wget -q https://github.com/princeton-vl/RAFT/releases/download/models/raft-things.pth
    wget -q https://github.com/princeton-vl/RAFT/releases/download/models/raft-sintel.pth
    cd ../..
fi

# إنشاء مجلدات مطلوبة
echo "📁 إنشاء مجلدات العمل..."
mkdir -p videos/input
mkdir -p videos/output
mkdir -p logs

# تصدير متغيرات البيئة
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export FLASK_APP=ui/app.py
export FLASK_ENV=production

# بدء التطبيق
echo "🎯 تشغيل التطبيق على المنفذ 5000..."
echo "📱 يمكنك الوصول للتطبيق عبر: http://$(curl -s ifconfig.me):5000"

# تشغيل التطبيق باستخدام Gunicorn للإنتاج
if command -v gunicorn &> /dev/null; then
    echo "🚀 تشغيل مع Gunicorn..."
    cd ui
    gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 300 app:app
else
    echo "🐍 تشغيل مع Flask Dev Server..."
    python3 ui/app.py
fi