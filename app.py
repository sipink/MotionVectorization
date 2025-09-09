#!/usr/bin/env python3
"""
Motion Vectorization Control Panel
واجهة تحكم لمعالجة الفيديو باستخدام RunPod
"""

import os
import json
import datetime
import subprocess
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import paramiko
import threading
import time

app = Flask(__name__)
app.secret_key = 'motion_vectorization_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# إنشاء المجلدات المطلوبة
os.makedirs('uploads', exist_ok=True)
os.makedirs('downloads', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# إعدادات RunPod (سيتم تحديثها من الواجهة)
runpod_config = {
    'host': '',
    'port': 22,
    'username': 'root',
    'password': '',
    'ssh_key_path': '',
    'remote_path': '/workspace/motion_vectorization',
    'connected': False
}

# حالة المهام النشطة
active_jobs = {}

def allowed_file(filename):
    """التحقق من أن الملف فيديو مسموح"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_message(message):
    """كتابة رسالة في السجل"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry.strip())
    
    with open('logs/app.log', 'a', encoding='utf-8') as f:
        f.write(log_entry)

def test_runpod_connection():
    """اختبار الاتصال بـ RunPod"""
    if not runpod_config['host']:
        return False, "عنوان RunPod غير محدد"
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        if runpod_config['ssh_key_path'] and os.path.exists(runpod_config['ssh_key_path']):
            # استخدام SSH Key
            ssh.connect(
                runpod_config['host'],
                port=runpod_config['port'],
                username=runpod_config['username'],
                key_filename=runpod_config['ssh_key_path'],
                timeout=10
            )
        else:
            # استخدام كلمة المرور
            ssh.connect(
                runpod_config['host'],
                port=runpod_config['port'],
                username=runpod_config['username'],
                password=runpod_config['password'],
                timeout=10
            )
        
        # تنفيذ أمر بسيط للتأكد
        stdin, stdout, stderr = ssh.exec_command('echo "connected"')
        result = stdout.read().decode().strip()
        ssh.close()
        
        if result == "connected":
            runpod_config['connected'] = True
            return True, "الاتصال نجح!"
        else:
            return False, "فشل في التحقق من الاتصال"
            
    except Exception as e:
        runpod_config['connected'] = False
        return False, f"خطأ في الاتصال: {str(e)}"

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html', 
                         runpod_connected=runpod_config['connected'],
                         active_jobs=active_jobs)

@app.route('/config', methods=['GET', 'POST'])
def config():
    """صفحة إعداد RunPod"""
    if request.method == 'POST':
        runpod_config['host'] = request.form['host']
        runpod_config['port'] = int(request.form.get('port', 22))
        runpod_config['username'] = request.form.get('username', 'root')
        runpod_config['password'] = request.form.get('password', '')
        runpod_config['ssh_key_path'] = request.form.get('ssh_key_path', '')
        runpod_config['remote_path'] = request.form.get('remote_path', '/workspace/motion_vectorization')
        
        # حفظ الإعدادات
        with open('runpod_config.json', 'w') as f:
            json.dump(runpod_config, f)
        
        # اختبار الاتصال
        success, message = test_runpod_connection()
        if success:
            flash(f'تم حفظ الإعدادات بنجاح! {message}', 'success')
        else:
            flash(f'تم حفظ الإعدادات ولكن: {message}', 'warning')
        
        log_message(f"تم تحديث إعدادات RunPod: {message}")
        return redirect(url_for('config'))
    
    return render_template('config.html', config=runpod_config)

@app.route('/upload', methods=['POST'])
def upload_video():
    """رفع فيديو جديد"""
    if 'video' not in request.files:
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # إضافة timestamp لتجنب تضارب الأسماء
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        log_message(f"تم رفع الفيديو: {filename}")
        return jsonify({'success': True, 'filename': filename, 'filepath': filepath})
    
    return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """بدء معالجة الفيديو على RunPod"""
    data = request.json
    video_file = data.get('video_file')
    
    if not video_file:
        return jsonify({'error': 'لم يتم تحديد ملف الفيديو'}), 400
    
    if not runpod_config['connected']:
        success, message = test_runpod_connection()
        if not success:
            return jsonify({'error': f'فشل الاتصال بـ RunPod: {message}'}), 500
    
    # إنشاء مهمة جديدة
    job_id = f"job_{int(time.time())}"
    active_jobs[job_id] = {
        'video_file': video_file,
        'status': 'preparing',
        'progress': 0,
        'start_time': datetime.datetime.now().isoformat(),
        'log': []
    }
    
    # بدء المعالجة في خيط منفصل
    thread = threading.Thread(target=process_video_on_runpod, args=(job_id, video_file))
    thread.daemon = True
    thread.start()
    
    log_message(f"بدأت مهمة معالجة: {job_id} للفيديو: {video_file}")
    return jsonify({'job_id': job_id, 'status': 'started'})

def process_video_on_runpod(job_id, video_file):
    """معالجة الفيديو على RunPod"""
    try:
        job = active_jobs[job_id]
        job['log'].append("بدء الاتصال بـ RunPod...")
        job['status'] = 'uploading'
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # الاتصال
        if runpod_config['ssh_key_path'] and os.path.exists(runpod_config['ssh_key_path']):
            ssh.connect(
                runpod_config['host'],
                port=runpod_config['port'],
                username=runpod_config['username'],
                key_filename=runpod_config['ssh_key_path']
            )
        else:
            ssh.connect(
                runpod_config['host'],
                port=runpod_config['port'],
                username=runpod_config['username'],
                password=runpod_config['password']
            )
        
        # إنشاء SFTP لنقل الملفات
        sftp = ssh.open_sftp()
        
        # إنشاء المجلد البعيد إذا لم يكن موجوداً
        try:
            sftp.mkdir(runpod_config['remote_path'])
        except:
            pass
        
        try:
            sftp.mkdir(f"{runpod_config['remote_path']}/videos")
        except:
            pass
        
        # رفع الفيديو
        local_path = os.path.join('uploads', video_file)
        remote_video_path = f"{runpod_config['remote_path']}/videos/{video_file}"
        
        job['log'].append(f"رفع الفيديو: {video_file}")
        sftp.put(local_path, remote_video_path)
        job['progress'] = 20
        
        # رفع الكود إذا لم يكن موجوداً
        job['log'].append("التأكد من وجود الكود...")
        setup_code_on_runpod(ssh, sftp)
        job['progress'] = 30
        
        # بدء المعالجة
        job['status'] = 'processing'
        job['log'].append("بدء معالجة الفيديو...")
        
        # تشغيل المعالجة
        video_name = os.path.splitext(video_file)[0]
        cmd = f"cd {runpod_config['remote_path']} && echo '{video_file}' > videos/current.txt && ./scripts/script.sh videos/current.txt"
        
        stdin, stdout, stderr = ssh.exec_command(cmd)
        
        # مراقبة التقدم
        while True:
            line = stdout.readline()
            if not line:
                break
            job['log'].append(line.strip())
            
            # تحديث التقدم بناءً على الخطوات
            if "PREPROCESS" in line:
                job['progress'] = 40
            elif "RAFT" in line:
                job['progress'] = 60
            elif "CLUSTER" in line:
                job['progress'] = 70
            elif "TRACK" in line:
                job['progress'] = 80
            elif "OPTIM" in line:
                job['progress'] = 85
            elif "PROGRAM" in line:
                job['progress'] = 90
        
        # تحويل إلى SVG
        job['log'].append("تحويل إلى SVG...")
        svg_cmd = f"cd {runpod_config['remote_path']} && ./scripts/convert_to_svg.sh {video_name} 30"
        stdin, stdout, stderr = ssh.exec_command(svg_cmd)
        stdout.read()  # انتظار الانتهاء
        
        # تحميل النتائج
        job['status'] = 'downloading'
        job['log'].append("تحميل النتائج...")
        
        output_dir = f"{runpod_config['remote_path']}/motion_vectorization/outputs/{video_name}_None"
        local_output_dir = f"downloads/{job_id}"
        os.makedirs(local_output_dir, exist_ok=True)
        
        # تحميل الملفات المهمة
        important_files = ['motion_file.json', 'motion_file.svg', 'time_bank.pkl', 'shape_bank.pkl']
        for file in important_files:
            try:
                remote_file = f"{output_dir}/{file}"
                local_file = f"{local_output_dir}/{file}"
                sftp.get(remote_file, local_file)
                job['log'].append(f"تم تحميل: {file}")
            except Exception as e:
                job['log'].append(f"خطأ في تحميل {file}: {str(e)}")
        
        job['progress'] = 100
        job['status'] = 'completed'
        job['end_time'] = datetime.datetime.now().isoformat()
        job['output_dir'] = local_output_dir
        job['log'].append("تمت المعالجة بنجاح!")
        
        sftp.close()
        ssh.close()
        
    except Exception as e:
        job['status'] = 'error'
        job['log'].append(f"خطأ: {str(e)}")
        log_message(f"خطأ في معالجة {job_id}: {str(e)}")

def setup_code_on_runpod(ssh, sftp):
    """رفع الكود إلى RunPod إذا لم يكن موجوداً"""
    try:
        # التحقق من وجود المجلد
        sftp.stat(f"{runpod_config['remote_path']}/scripts")
        return  # الكود موجود
    except:
        pass
    
    # رفع الكود
    import tarfile
    
    # إنشاء أرشيف مضغوط للكود
    with tarfile.open('code_archive.tar.gz', 'w:gz') as tar:
        tar.add('motion_vectorization', arcname='motion_vectorization')
        tar.add('RAFT', arcname='RAFT')
        tar.add('scripts', arcname='scripts')
        tar.add('svg_utils', arcname='svg_utils')
    
    # رفع الأرشيف
    sftp.put('code_archive.tar.gz', f"{runpod_config['remote_path']}/code_archive.tar.gz")
    
    # استخراج الأرشيف
    stdin, stdout, stderr = ssh.exec_command(f"cd {runpod_config['remote_path']} && tar -xzf code_archive.tar.gz")
    stdout.read()
    
    # جعل السكريبت قابل للتنفيذ
    ssh.exec_command(f"chmod +x {runpod_config['remote_path']}/scripts/*.sh")
    
    # حذف الأرشيف المحلي
    os.remove('code_archive.tar.gz')

@app.route('/job_status/<job_id>')
def job_status(job_id):
    """الحصول على حالة المهمة"""
    if job_id in active_jobs:
        return jsonify(active_jobs[job_id])
    return jsonify({'error': 'المهمة غير موجودة'}), 404

@app.route('/download_result/<job_id>/<filename>')
def download_result(job_id, filename):
    """تحميل ملف من نتائج المهمة"""
    if job_id not in active_jobs or active_jobs[job_id]['status'] != 'completed':
        return jsonify({'error': 'المهمة غير مكتملة'}), 404
    
    output_dir = active_jobs[job_id].get('output_dir')
    if not output_dir:
        return jsonify({'error': 'مجلد النتائج غير موجود'}), 404
    
    file_path = os.path.join(output_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    
    return jsonify({'error': 'الملف غير موجود'}), 404

@app.route('/test_connection', methods=['POST'])
def test_connection_endpoint():
    """اختبار الاتصال بـ RunPod عبر AJAX"""
    data = request.json
    
    # تحديث الإعدادات مؤقتاً لاختبار الاتصال
    temp_config = runpod_config.copy()
    if data:
        temp_config.update({
            'host': data.get('host', ''),
            'port': int(data.get('port', 22)),
            'username': data.get('username', 'root'),
            'password': data.get('password', ''),
            'ssh_key_path': data.get('ssh_key_path', ''),
            'remote_path': data.get('remote_path', '/workspace/motion_vectorization')
        })
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        if temp_config['ssh_key_path'] and os.path.exists(temp_config['ssh_key_path']):
            ssh.connect(
                temp_config['host'],
                port=temp_config['port'],
                username=temp_config['username'],
                key_filename=temp_config['ssh_key_path'],
                timeout=15
            )
        else:
            ssh.connect(
                temp_config['host'],
                port=temp_config['port'],
                username=temp_config['username'],
                password=temp_config['password'],
                timeout=15
            )
        
        # تنفيذ أوامر اختبار
        commands = [
            'echo "connection_test"',
            'nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1',
            f'ls {temp_config["remote_path"]} 2>/dev/null || echo "directory_not_found"',
            'python3 --version 2>/dev/null || python --version 2>/dev/null || echo "python_not_found"'
        ]
        
        results = {}
        for i, cmd in enumerate(commands):
            stdin, stdout, stderr = ssh.exec_command(cmd)
            output = stdout.read().decode().strip()
            error = stderr.read().decode().strip()
            
            if i == 0:  # connection test
                results['connection'] = output == "connection_test"
            elif i == 1:  # GPU info
                results['gpu'] = output if output and not error else "لم يتم العثور على GPU"
            elif i == 2:  # directory check
                results['directory'] = "موجود" if output != "directory_not_found" else "غير موجود"
            elif i == 3:  # Python version
                results['python'] = output if "python_not_found" not in output else "غير مثبت"
        
        ssh.close()
        
        return jsonify({
            'success': True,
            'message': 'تم الاتصال بنجاح!',
            'details': results
        })
        
    except paramiko.AuthenticationException:
        return jsonify({
            'success': False,
            'message': 'فشل في المصادقة - تحقق من اسم المستخدم وكلمة المرور'
        }), 401
    except paramiko.SSHException as e:
        return jsonify({
            'success': False,
            'message': f'خطأ SSH: {str(e)}'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'خطأ في الاتصال: {str(e)}'
        }), 500

@app.route('/logs')
def view_logs():
    """عرض سجل التطبيق"""
    try:
        with open('logs/app.log', 'r', encoding='utf-8') as f:
            logs = f.read()
        return render_template('logs.html', logs=logs)
    except:
        return render_template('logs.html', logs="لا توجد سجلات بعد")

if __name__ == '__main__':
    # تحميل الإعدادات المحفوظة
    try:
        with open('runpod_config.json', 'r') as f:
            saved_config = json.load(f)
            runpod_config.update(saved_config)
        log_message("تم تحميل إعدادات RunPod المحفوظة")
    except:
        log_message("لم يتم العثور على إعدادات محفوظة")
    
    log_message("بدء تشغيل واجهة التحكم...")
    app.run(host='0.0.0.0', port=5000, debug=True)