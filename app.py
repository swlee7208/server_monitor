from flask import Flask, render_template, jsonify
import psutil
import subprocess

app = Flask(__name__)

def get_gpu_usage():
    try:
        output = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits']
        ).decode('utf-8').strip()

        gpu_list = []
        for line in output.split('\n'):
            gpu_util, gpu_temp, mem_used, mem_total = map(int, line.split(', '))
            gpu_list.append({
                'gpu_util': gpu_util,
                'gpu_temp': gpu_temp,
                'mem_used': mem_used,
                'mem_total': mem_total
            })
        return gpu_list
    except Exception:
        return []

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if 'coretemp' in temps:
            return temps['coretemp'][0].current
        elif 'cpu-thermal' in temps:
            return temps['cpu-thermal'][0].current
    except Exception:
        return None
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_temp = get_cpu_temp()
    ram = psutil.virtual_memory()
    gpus = get_gpu_usage()

    return jsonify({
        'cpu': cpu_percent,
        'cpu_temp': cpu_temp,
        'ram_used': ram.used // (1024**2),
        'ram_total': ram.total // (1024**2),
        'gpus': gpus
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


@app.route('/cuda')
def cuda_status():
    return "<h2>Alpha CUDA Simulation Status Page (Coming Soon)</h2>"

