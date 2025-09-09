#!/bin/bash

# سكريپت ضغط المشروع للنقل إلى RunPod
echo "📦 بدء ضغط ملفات المشروع..."

# إنشاء مجلد مؤقت للملفات المطلوبة
mkdir -p runpod_package

# نسخ الملفات الأساسية
echo "📋 نسخ الملفات الأساسية..."
cp -r RAFT runpod_package/
cp -r motion_vectorization runpod_package/
cp -r svg_utils runpod_package/
cp -r scripts runpod_package/
cp -r ui runpod_package/
cp -r videos runpod_package/

# نسخ ملفات التكوين
cp pyproject.toml runpod_package/
cp requirements.txt runpod_package/
cp start_runpod.sh runpod_package/
cp runpod_deployment_guide.md runpod_package/
cp docker_alternative.md runpod_package/

# حذف الملفات غير المطلوبة
echo "🧹 تنظيف الملفات غير المطلوبة..."
find runpod_package -name "*.pyc" -delete
find runpod_package -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find runpod_package -name ".git" -type d -exec rm -rf {} + 2>/dev/null

# ضغط المجلد
echo "🗜️  ضغط الملفات..."
tar -czf motion_vectorization_runpod.tar.gz runpod_package/

# حذف المجلد المؤقت
rm -rf runpod_package

# عرض معلومات الملف المضغوط
file_size=$(du -h motion_vectorization_runpod.tar.gz | cut -f1)
echo "✅ تم إنشاء الملف المضغوط: motion_vectorization_runpod.tar.gz"
echo "📏 حجم الملف: $file_size"
echo ""
echo "📤 الآن يمكنك نقل الملف إلى RunPod باستخدام:"
echo "scp motion_vectorization_runpod.tar.gz root@YOUR_POD_IP:/workspace/"
echo ""
echo "📂 ثم فك الضغط على RunPod:"
echo "cd /workspace && tar -xzf motion_vectorization_runpod.tar.gz"
echo "cd runpod_package && ./start_runpod.sh"