
from flask import Flask, render_template, jsonify, request, redirect, url_for
import psutil
import subprocess
import os
import re
import time
import shlex
from pathlib import Path
from collections import deque
from datetime import datetime
from decimal import Decimal

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

app = Flask(__name__)

# =============================
# Config by ENV
# =============================
DB_HOST   = os.getenv('ALPHA_DB_HOST', os.getenv('PGHOST', '127.0.0.1'))
DB_PORT   = int(os.getenv('ALPHA_DB_PORT', os.getenv('PGPORT', '5432')))
DB_NAME   = os.getenv('ALPHA_DB_NAME', os.getenv('PGDATABASE', 'stock'))
DB_USER   = os.getenv('ALPHA_DB_USER', os.getenv('PGUSER', 'postgres'))
DB_PASS   = os.getenv('ALPHA_DB_PASS', os.getenv('PGPASSWORD', 'db4woo'))
# cuda 테이블이 alpha 스키마에 있는 경우가 많아서 기본값을 alpha로 둠
DB_SCHEMA = os.getenv('ALPHA_DB_SCHEMA', 'alpha')

# 테이블명 커스터마이즈 가능
SIMUL_TABLE  = os.getenv('ALPHA_SIMUL_TABLE',  'alpha_cuda_simul')
RESULT_TABLE = os.getenv('ALPHA_RESULT_TABLE', 'cuda_simul_result')
PARAM_TABLE = os.getenv('ALPHA_PARAM_TABLE',   'simul_param_set')

ENV_TABLE = os.getenv('ALPHA_ENV_TABLE', 'alpha_env')
APPLY_ENABLED = os.getenv('ALPHA_ENABLE_APPLY', '1')

valid_ident = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')  # 컬럼명 검증용

# ===== Alpha CUDA config =====
ALPHA_CUDA_BIN   = '/home/alpha/bin/alpha_cuda'     # 요구사항 1
ALPHA_CUDA_CWD   = '/home/alpha'                    # 필요시 원하는 작업 디렉터리
ALPHA_LOG_DIR    = '/home/alpha/alpha_logs'         # 로그 디렉터리 (요구사항 3)
ALPHA_TMUX_PREF  = 'ac'                             # tmux 세션 prefix


# 상세에서의 별칭 기준(둘 중 하나만 있어도 됨 → alias로 통일)
CANONICAL_MIN = {
    'simul_id': ['simul_id'],
    'gid':      ['gid', 'case_id'],
    'wins':     ['wins', 'win_cnt'],
    'trades':   ['trades', 'trade_cnt'],
}


# ====== Simulation 컬럼 사양 ======
# 리스트(하단 테이블)에 보여줄 컬럼
SIMUL_LIST_EXPECTED = [
    'simul_id',
    'simul_type',
    'simul_desc',
    'main_simul_id',
    'param_set_id',
    'main_simul_gid',
    'simul_name',
    'simul_from_date',
    'simul_to_date',
    'case_cnt',
    'start_time',
    'end_time',
    'progress',
    'simul_candle',
    'reg_time',
    'reason',
    'is_active',
    'status'
]

# 쓰기(등록/수정) 허용 컬럼 (요청한 입력 항목)
SIMUL_WRITE_FIELDS = [
    'simul_id',
    'simul_name',
    'simul_type',
    'reg_time',
    'simul_from_date',
    'simul_to_date',
    'param_set_id',
    'is_active',
    'status',
    'main_simul_id',
    'main_simul_gid',
    'simul_candle',
    'simul_desc',
    'reason',
    'status',
    'reason'
]



# =========[ helpers: Decimal cnt / total cases ]=========
from decimal import Decimal, getcontext
getcontext().prec = 28

def _calc_data_cnt(minv, maxv, step) -> int:
    """[data_cnt] = floor((max - min) / step) + 1, Decimal로 안전 계산"""
    if minv is None or maxv is None or step is None:
        return 0
    a = Decimal(str(minv))
    b = Decimal(str(maxv))
    s = Decimal(str(step))
    if s <= 0 or a > b:
        return 0
    # 0.1, 0.01 등 이진표현 오차 방지 위해 Decimal 사용
    n = (b - a) / s
    # 음수/실수 입력을 방어적으로 보정
    n_int = int(n.to_integral_value(rounding='ROUND_FLOOR'))
    return n_int + 1

def _fetch_paramset_rows(set_id: int):
    with get_pg_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        sql = f'''
            SELECT set_id, param_no, param_name, param_desc, data_type,
                   data_min, data_max, data_step_size, data_cnt, is_active
            FROM {qualify(PARAM_TABLE)}
            WHERE set_id = %s
            ORDER BY param_no ASC
        '''
        cur.execute(sql, (set_id,))
        rows = cur.fetchall()

    # total_cases = 활성화된 data_cnt의 곱
    total_cases = 0
    prod = 1
    active_any = False
    for r in rows:
        if int(r.get('is_active', 0)) == 1:
            c = int(r.get('data_cnt') or 0)
            active_any = True
            prod *= max(c, 0)
    if active_any:
        total_cases = prod
    return rows, total_cases


# ---- 날짜 파서 (timestamp 컬럼용) ----
from datetime import datetime

def _parse_ts(v):
    """입력값 v를 timestamp로 파싱. 'yyyyMMddHHmmss' 또는 ISO-8601 허용."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    # 14자리(yyyyMMddHHmmss)
    if len(s) == 14 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d%H%M%S")
    # ISO 시도 (yyyy-mm-dd hh:mm:ss 등)
    try:
        # 3.11 이하 호환: fromisoformat 실패 시 strptime 재시도
        return datetime.fromisoformat(s.replace(' ', 'T'))
    except Exception:
        pass
    # fallback common patterns
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    raise ValueError(f"Invalid timestamp format: {s}")

def _clean_simul_payload(raw: dict) -> dict:
    """입력 payload 정리/검증: 허용 필드만 추출 + 형식 변환"""
    data = {}
    for k in SIMUL_WRITE_FIELDS:
        if k not in raw:
            continue
        v = raw[k]
        if k in ('simul_id', 'simul_candle', 'simul_type','simul_candle',
                 'main_simul_id', 'main_simul_gid', 'param_set_id', 'is_active'):
            data[k] = None if v is None or str(v).strip() == '' else int(v)
        elif k in ('simul_from_date', 'simul_to_date', 'reg_time'):
            data[k] = _parse_ts(v)
        elif k in ('simul_name', 'simul_desc', 'reason' ):
            data[k] = None if v is None else str(v).strip()
        else:
            data[k] = v
    return data

def _filter_existing_cols(conn, table_name: str, payload: dict) -> dict:
    cols = get_table_columns(conn, table_name)
    return {k: v for k, v in payload.items() if k in cols}



# =============================
# Monitor helpers
# =============================

def _tmux_has_session(name: str) -> bool:
    try:
        return subprocess.run(['tmux', 'has-session', '-t', name],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False

def _tmux_list_sessions(prefix: str):
    try:
        out = subprocess.check_output(['tmux', 'list-sessions', '-F', '#{session_name}\t#{session_created}'],
                                      stderr=subprocess.DEVNULL).decode()
        rows = []
        for line in out.splitlines():
            if not line.strip():
                continue
            name, created = line.split('\t', 1)
            if name.startswith(prefix + '-'):
                rows.append({'session': name, 'created': int(created)})
        rows.sort(key=lambda r: r['created'])
        return rows
    except Exception:
        return []


def _extract_slug(opts_list):
    """옵션에서 -s/--simul-id, -e/--env-no 같은 걸 슬러그로 조금 붙여줌(없어도 괜찮음)."""
    slug = []
    for i, tok in enumerate(opts_list):
        if tok in ('-s', '--simul-id') and i + 1 < len(opts_list):
            slug.append(f"s{opts_list[i+1]}")
        if tok in ('-e', '--env-no') and i + 1 < len(opts_list):
            slug.append(f"e{opts_list[i+1]}")
    return ('-' + '-'.join(slug)) if slug else ''


def _new_session_name(opts_list):
    ts = time.strftime('%y%m%d-%H%M%S')
    return f"{ALPHA_TMUX_PREF}-{ts}{_extract_slug(opts_list)}"  # 예: ac-250824-101512-s6-e5

def _ensure_log_dir():
    Path(ALPHA_LOG_DIR).mkdir(parents=True, exist_ok=True)

def _log_path_for_session(session: str) -> str:
    return str(Path(ALPHA_LOG_DIR) / f"{session}.log")

def _touch_symlink_latest(target_log: str):
    # latest.log 심볼릭 링크를 최근 실행 로그로 갱신
    latest = Path(ALPHA_LOG_DIR) / 'latest.log'
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
    except Exception:
        pass
    try:
        latest.symlink_to(Path(target_log).name)  # 같은 폴더이므로 상대 링크
    except Exception:
        # 심볼릭 링크 불가한 FS면 그냥 복사 없이 무시
        pass

def _tmux_launch_with_opts(opts_str: str):
    _ensure_log_dir()

    # 옵션 파싱(사용자가 입력한 옵션 그대로, 공백/따옴표 안전하게 처리)
    try:
        opts_list = shlex.split(opts_str or '')
    except Exception as e:
        return {'ok': False, 'msg': f'option parse error: {e}'}

    session = _new_session_name(opts_list)
    log_path = _log_path_for_session(session)

    # 이미 동일 이름이 있으면(희박하지만) 뒤에 -1 붙이는 처리
    if _tmux_has_session(session):
        k = 1
        while _tmux_has_session(f"{session}-{k}"):
            k += 1
        session = f"{session}-{k}"

    try:
        # tmux new-session -d -s session -c CWD BIN ARGS...
        cmd = ['tmux', 'new-session', '-d', '-s', session, '-c', ALPHA_CUDA_CWD, ALPHA_CUDA_BIN] + opts_list
        subprocess.check_call(cmd)

        # pipe-pane으로 세션 출력 → 로그파일 append
        subprocess.check_call(['tmux', 'pipe-pane', '-o', '-t', session, f'cat >> {log_path}'])

        _touch_symlink_latest(log_path)
        return {'ok': True, 'session': session, 'log': log_path}
    except subprocess.CalledProcessError as e:
        return {'ok': False, 'msg': f'launch failed: {e}'}

def _tmux_kill(session: str):
    try:
        if not _tmux_has_session(session):
            return {'ok': True, 'msg': f'session "{session}" not running'}
        subprocess.check_call(['tmux', 'kill-session', '-t', session])
        return {'ok': True}
    except subprocess.CalledProcessError as e:
        return {'ok': False, 'msg': f'kill failed: {e}'}

def _tmux_status_list():
    rows = _tmux_list_sessions(ALPHA_TMUX_PREF)
    # pid(있으면) 조회
    for r in rows:
        try:
            out = subprocess.check_output(['tmux', 'list-panes', '-t', r['session'], '-F', '#{pane_pid}']).decode().strip()
            r['pid'] = int(out) if out and out.isdigit() else None
        except Exception:
            r['pid'] = None
        r['log'] = _log_path_for_session(r['session'])
    return rows


@app.route('/api/alpha_cuda/launch', methods=['POST'])
def api_alpha_cuda_launch():
    js = request.get_json(silent=True) or {}
    opts = js.get('opts', '')              # 예: "-s 6 -f 3400"
    res = _tmux_launch_with_opts(opts)
    return (jsonify(res), 200 if res.get('ok') else 500)

@app.route('/api/alpha_cuda/stop', methods=['POST'])
def api_alpha_cuda_stop():
    js = request.get_json(silent=True) or {}
    session = js.get('session')            # 특정 세션만 종료
    if not session:
        return jsonify({'ok': False, 'msg': 'session required'}), 400
    res = _tmux_kill(session)
    return (jsonify(res), 200 if res.get('ok') else 500)

@app.route('/api/alpha_cuda/sessions')
def api_alpha_cuda_sessions():
    return jsonify(_tmux_status_list())

@app.route('/api/alpha_cuda/help')
def api_alpha_cuda_help():
    return jsonify(_tmux_status_list())

@app.route('/api/alpha_cuda/log')
def api_alpha_cuda_log():
    # ?session=ac-...&lines=100  (없으면 latest.log)
    lines = request.args.get('lines', default=12, type=int)
    lines = max(1, min(lines, 1000))
    session = request.args.get('session')
    if session:
        path = _log_path_for_session(session)
    else:
        path = str(Path(ALPHA_LOG_DIR) / 'latest.log')
    try:
        data = tail_file(path, n=lines)
        return jsonify({'path': path, 'lines': data})
    except Exception as e:
        return jsonify({'path': path, 'lines': [], 'error': str(e)}), 500


@app.route('/api/alpha_cuda/log/select', methods=['POST'])
def api_alpha_cuda_log_select():
    js = request.get_json(silent=True) or {}
    session = (js.get('session') or '').strip()
    if not session:
        return jsonify({'ok': False, 'msg': 'session required'}), 400

    log_path = _log_path_for_session(session)
    p = Path(log_path)
    if not p.exists():
        return jsonify({'ok': False, 'msg': f'log not found: {log_path}'}), 404

    latest = Path(ALPHA_LOG_DIR) / 'latest.log'
    try:
        # 이미 같은 대상이면 작업 스킵
        if latest.is_symlink():
            try:
                if latest.resolve() == p.resolve():
                    return jsonify({'ok': True, 'session': session, 'log': str(p)})
            except Exception:
                pass

        # 임시 링크를 만든 뒤 원자적으로 교체 → 레이스/존재 에러 회피
        tmp = Path(ALPHA_LOG_DIR) / '.latest.tmp'
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

        # 절대 경로로 링크(신뢰성↑)
        tmp.symlink_to(p)
        os.replace(tmp, latest)    # atomic replace (기존 latest가 있어도 OK)

        return jsonify({'ok': True, 'session': session, 'log': str(p)})
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'symlink failed: {e}'}), 500


# === progress by simul_id(s) ===
@app.get('/api/simul/progress')
def api_simul_progress():
    """
    /api/simul/progress?ids=7,12,15
    -> {"7": 42.3, "12": 88.1, "15": 100.0}
    """
    ids_param = (request.args.get('ids') or '').strip()
    ids = [int(x) for x in ids_param.split(',') if x.isdigit()]
    if not ids:
        return jsonify({})

    try:
        with get_pg_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # progress가 numeric/float/decimal 어떤 타입이든 double로 캐스팅
            sql = f"SELECT simul_id, (progress)::double precision AS progress FROM {qualify(SIMUL_TABLE)} WHERE simul_id = ANY(%s)"
            cur.execute(sql, (ids,))
            rows = cur.fetchall()
        return jsonify({str(r['simul_id']): (float(r['progress']) if r['progress'] is not None else None) for r in rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


############################################



def tail_file(path, n=10):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            dq = deque(f, maxlen=n)
        return [line.rstrip('\n') for line in dq]

    except Exception:
        return []

def get_gpu_usage():
    try:
        output = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ]
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
        if not temps:
            return None
        # 우선순위 키 후보
        for key in ("coretemp", "k10temp", "cpu-thermal", "acpitz"):
            if key in temps and temps[key]:
                return temps[key][0].current
        # fallback: 첫번째 센서
        first_key = next(iter(temps))
        return temps[first_key][0].current
    except Exception:
        return None



# === ADD: Globals for rate calculations (put near other monitor helpers) ===
_io_prev = {
    'time': None,
    'disk': None,   # psutil._common.sdiskio
    'net': None     # psutil._common.snetio
}

def _rate_per_sec(new, old, dt):
    return (new - old) / dt if (old is not None and dt > 0) else 0.0

def _mb(x):
    return x / (1024.0 * 1024.0)

def get_disk_net_rates():
    """Return (disk, net) rates over time using psutil counters.
    disk: {read_MBps, write_MBps, read_iops, write_iops}
    net:  {up_MBps, down_MBps}
    """
    now = time.time()
    d = psutil.disk_io_counters()
    n = psutil.net_io_counters()

    prev_t = _io_prev['time']
    prev_d = _io_prev['disk']
    prev_n = _io_prev['net']

    if prev_t is None:
        _io_prev.update({'time': now, 'disk': d, 'net': n})
        return (
            {'read_MBps': 0.0, 'write_MBps': 0.0, 'read_iops': 0.0, 'write_iops': 0.0},
            {'up_MBps': 0.0, 'down_MBps': 0.0}
        )

    dt = max(1e-6, now - prev_t)

    disk_rates = {
        'read_MBps':  _mb(_rate_per_sec(d.read_bytes,  prev_d.read_bytes,  dt)),
        'write_MBps': _mb(_rate_per_sec(d.write_bytes, prev_d.write_bytes, dt)),
        'read_iops':  _rate_per_sec(d.read_count,      prev_d.read_count,  dt),
        'write_iops': _rate_per_sec(d.write_count,     prev_d.write_count, dt),
    }
    net_rates = {
        'up_MBps':   _mb(_rate_per_sec(n.bytes_sent, prev_n.bytes_sent, dt)),
        'down_MBps': _mb(_rate_per_sec(n.bytes_recv, prev_n.bytes_recv, dt)),
    }

    _io_prev.update({'time': now, 'disk': d, 'net': n})
    return (disk_rates, net_rates)


def get_gpu_extra():
    """Query power, clocks, fan via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=power.draw,clocks.gr,clocks.mem,fan.speed',
                '--format=csv,noheader,nounits'
            ]
        ).decode('utf-8').strip()
        res = []
        for line in out.split('\n'):
            p, cgr, cmem, fan = [x.strip() for x in line.split(',')]
            res.append({
                'power_w': float(p),
                'clock_gr': int(cgr),
                'clock_mem': int(cmem),
                'fan_pct': float(fan)
            })
        return res
    except Exception:
        return []


def get_cpu_fans():
    fans = []

    # 1) psutil
    try:
        data = psutil.sensors_fans()
        for name, arr in data.items():
            for f in arr:
                rpm = getattr(f, 'current', None)
                if rpm is not None:
                    fans.append({
                        'sensor': name,
                        'label': f.label or '',
                        'rpm': int(rpm)
                    })
    except Exception:
        pass

    if fans:
        return fans

    # 2) /sys/class/hwmon (Linux)
    try:
        base = '/sys/class/hwmon'
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                # chip name
                chip = ''
                try:
                    with open(os.path.join(root, 'name'), 'r') as nf:
                        chip = nf.read().strip()
                except Exception:
                    chip = os.path.basename(root)

                # fanX_input files
                for fn in sorted(f for f in files if re.match(r'^fan\d+_input$', f)):
                    path = os.path.join(root, fn)
                    try:
                        with open(path, 'r') as ff:
                            rpm = int(ff.read().strip() or '0')
                        label_path = os.path.join(root, fn.replace('_input', '_label'))
                        label = ''
                        if os.path.exists(label_path):
                            with open(label_path, 'r') as lf:
                                label = lf.read().strip()
                        fans.append({
                            'sensor': chip,
                            'label': label or fn.replace('_input', ''),
                            'rpm': rpm
                        })
                    except Exception:
                        continue
    except Exception:
        pass

    if fans:
        return fans

    # 3) IPMI (server/BMC)
    try:
        out = subprocess.check_output(
            ['ipmitool', 'sdr', 'type', 'Fan'],
            timeout=2
        ).decode('utf-8', errors='ignore')
        for line in out.splitlines():
            # e.g. "FAN1 | 1400 RPM | ok"
            m = re.match(r'\s*([^|]+)\|\s*([0-9]+)\s*RPM', line)
            if m:
                label = m.group(1).strip()
                rpm = int(m.group(2))
                fans.append({
                    'sensor': 'ipmi',
                    'label': label,
                    'rpm': rpm
                })
    except Exception:
        pass

    return fans





def get_top_processes(topn=10):
    """Return top processes by CPU and Memory."""
    procs = []
    for p in psutil.process_iter(attrs=['pid', 'name']):
        try:
            cpu = p.cpu_percent(interval=0.0)
            mem = p.memory_info().rss
            procs.append({'pid': p.pid, 'name': p.info['name'] or '', 'cpu': cpu, 'mem_mb': _mb(mem)})
        except Exception:
            continue
    # quick warm-up for cpu_percent
    time.sleep(0.05)
    for item in procs:
        try:
            p = psutil.Process(item['pid'])
            item['cpu'] = p.cpu_percent(interval=0.0)
        except Exception:
            pass
    by_cpu = sorted(procs, key=lambda x: x['cpu'], reverse=True)[:topn]
    by_mem = sorted(procs, key=lambda x: x['mem_mb'], reverse=True)[:topn]
    return {'top_cpu': by_cpu, 'top_mem': by_mem}


def get_gpu_processes():
    """Use nvidia-smi to list GPU processes (PID, name, used_memory MB)."""
    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits'
            ]
        ).decode('utf-8').strip()
        rows = []
        if not out:
            return rows
        for line in out.split('\n'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 3:
                pid = int(parts[0])
                name = parts[1]
                mem_mb = float(parts[2])  # already MB
                rows.append({'pid': pid, 'name': name, 'vram_mb': mem_mb})
        return rows
    except Exception:
        return []




# =============================
# DB helpers
# =============================
def get_pg_conn():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Run: pip install psycopg2-binary")
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS,
        connect_timeout=5, keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=3
    )

def qualify(table_name: str) -> str:
    # "alpha"."cuda_simul_result" 같은 형식으로 안전하게
    return f'"{DB_SCHEMA}"."{table_name}"'

def get_table_columns(conn, table_name: str):
    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (DB_SCHEMA, table_name))
        return {row[0] for row in cur.fetchall()}

def _alias_if_exists(cols, options, alias):
    """options 중 존재하는 첫 원본 컬럼을 alias로 선택."""
    for c in options:
        if c in cols:
            return f'{c} AS {alias}'
    return None

# =============================
# Query helpers
# =============================
def fetch_simul_list(limit=200):
    with get_pg_conn() as conn:
        cols = get_table_columns(conn, SIMUL_TABLE)  # 테이블 실제 컬럼 집합
        present = [c for c in SIMUL_LIST_EXPECTED if c in cols]

        if 'simul_id' not in present:
            raise RuntimeError(f'[{DB_SCHEMA}.{SIMUL_TABLE}] must contain simul_id. Existing cols={sorted(list(cols))}')

        # simul_id는 명시 alias로
        select_cols = []
        for c in present:
            if c == 'simul_id':
                select_cols.append('simul_id AS simul_id')
            else:
                select_cols.append(c)
        select_list = ', '.join(select_cols)

        # 정렬 우선순위: end_time > reg_time > simul_id
        order_prefix = ''
        if 'end_time' in cols:
            order_prefix = 'end_time DESC NULLS LAST,'
        elif 'reg_time' in cols:
            order_prefix = 'reg_time DESC NULLS LAST,'

        #sql = f'''
            #SELECT {select_list}
            #FROM {qualify(SIMUL_TABLE)}
            #ORDER BY
                #{order_prefix}
                #simul_id DESC
            #LIMIT %s
        #'''

        sql = f'''
            SELECT {select_list}
            FROM {qualify(SIMUL_TABLE)}
            ORDER BY
                simul_id DESC
            LIMIT %s
        '''
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

    # 날짜/시간 문자열화
    for r in rows:
        for k in ('simul_from_date', 'simul_to_date', 'end_time', 'reg_time','start_time', 'end_time'):
            if k in r and r[k] is not None:
                if isinstance(r[k], datetime):
                    #r[k] = r[k].strftime('%Y-%m-%d %H:%M') if k in ('start_time', 'end_time') else r[k].strftime('%Y-%m-%d')
                    r[k] = r[k].strftime('%Y-%m-%d %H:%M')
                else:
                    r[k] = str(r[k])
    return rows



def fetch_simul_detail(simul_id: int, limit=500):
    with get_pg_conn() as conn:
        cols = get_table_columns(conn, RESULT_TABLE)

        # 필수(별칭) 준비
        sel_min = []
        for alias, options in CANONICAL_MIN.items():
            piece = _alias_if_exists(cols, options, alias)
            if piece:
                sel_min.append(piece)
        missing_aliases = [a for a in CANONICAL_MIN if not any(a in s for s in sel_min)]
        if missing_aliases:
            raise RuntimeError(
                f'[{DB_SCHEMA}.{RESULT_TABLE}] missing required columns for {missing_aliases}. Existing cols={sorted(list(cols))}'
            )

        # 선택 컬럼: total_profit 유사 + p{i}_name/val
        select_opt = []
        tp = _alias_if_exists(cols, ['total_profit', 'total_profit_ptr', 'profit_ptr'], 'total_profit')
        if tp:
            select_opt.append(tp)
        for i in range(1, 37):
            nk, vk = f'p{i}_name', f'p{i}_val'
            if nk in cols: select_opt.append(nk)
            if vk in cols: select_opt.append(vk)

        select_list = ', '.join(sel_min + select_opt) if select_opt else ', '.join(sel_min)

        # 승율 정렬
        sql = f'''
            SELECT {select_list}
            FROM {qualify(RESULT_TABLE)}
            WHERE simul_id = %s
            ORDER BY
                CASE
                    WHEN trades IS NOT NULL AND trades > 0 THEN (wins::float / trades) * 100.0
                    ELSE -1
                END DESC,
                {"total_profit DESC NULLS LAST," if "total_profit" in (tp or "") else ""}
                gid ASC
            LIMIT %s
        '''

        # 수익포인트 정렬
        sql2 = f"""
            SELECT {select_list}
            FROM {qualify(RESULT_TABLE)}
            WHERE simul_id = %s
            ORDER BY
                (total_profit)::double precision DESC NULLS LAST,
                gid ASC
            LIMIT %s
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql2, (simul_id, limit))
            rows = cur.fetchall()


    # 후처리: Decimal → float, win_rate, param_summary
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, Decimal):
                r[k] = float(v)

        tc = r.get('trades') or 0
        wc = r.get('wins') or 0
        r['win_rate'] = round((wc * 100.0) / tc, 2) if tc > 0 else None

        # 이름이 있고 값이 None이 아니면 포함 (0도 포함)
        params = []
        for i in range(1, 37):
            nk, vk = f'p{i}_name', f'p{i}_val'
            name = r.get(nk, None)
            val  = r.get(vk, None)
            if name is not None and str(name).strip() != '' and val is not None:
                params.append(f"{name}={val}")
        r['param_summary'] = "; ".join(params) if params else None

    return rows


# =============================
# Detail query (filters + pagination)
# =============================
def _resolve_result_columns(cols):
    """Return tuple: (gid_col, wins_col, trades_col, profit_expr)
    profit_expr is a dynamic SQL snippet using existing profit-like columns.
    """
    # gid / wins / trades: pick first existing from canonical options
    def first_present(options):
        for c in options:
            if c in cols:
                return c
        return None

    gid_col    = first_present(CANONICAL_MIN['gid'])
    wins_col   = first_present(CANONICAL_MIN['wins'])
    trades_col = first_present(CANONICAL_MIN['trades'])

    # profit columns that may exist
    profit_candidates = [c for c in ('total_profit', 'total_profit_ptr', 'profit_ptr') if c in cols]
    if profit_candidates:
        if len(profit_candidates) == 1:
            profit_expr = f"({profit_candidates[0]})::double precision"
        else:
            joined = ", ".join(profit_candidates)
            profit_expr = f"COALESCE({joined})::double precision"
    else:
        # No profit-like columns; use NULL
        profit_expr = "NULL::double precision"

    return gid_col, wins_col, trades_col, profit_expr


def fetch_simul_detail_paged(simul_id: int, *, page:int=1, page_size:int=200,
                             win_min=None, win_max=None,
                             profit_min=None, profit_max=None,
                             trades_min=None, trades_max=None,
                             sort_by:str='profit', sort_dir:str='desc'):
    """
    Return dict: {items: [...], total: N, page: x, page_size: y, pages: k}
    - sort_by in {'profit','win_rate'}
    - sort_dir in {'asc','desc'}
    """
    if page < 1: page = 1
    page_size = max(1, min(page_size, 5000))
    sort_by = (sort_by or 'profit').lower()
    if sort_by not in ('profit','win_rate'):
        sort_by = 'profit'
    sort_dir = (sort_dir or 'desc').lower()
    if sort_dir not in ('asc','desc'):
        sort_dir = 'desc'

    with get_pg_conn() as conn:
        cols = get_table_columns(conn, RESULT_TABLE)
        gid_col, wins_col, trades_col, profit_expr = _resolve_result_columns(cols)
        if not (gid_col and wins_col and trades_col):
            raise RuntimeError(f"[{DB_SCHEMA}.{RESULT_TABLE}] missing required gid/wins/trades columns; existing={sorted(list(cols))}")

        # Build select columns (same as legacy + computed win_rate)
        sel_min = [
            f"{gid_col} AS gid",
            f"{wins_col} AS wins",
            f"{trades_col} AS trades",
            f"{profit_expr} AS total_profit"
        ]
        # add p{i}_name/val if present
        for i in range(1, 37):
            nk, vk = f"p{i}_name", f"p{i}_val"
            if nk in cols: sel_min.append(nk)
            if vk in cols: sel_min.append(vk)

        # computed win_rate expression
        win_rate_expr = f"""CASE
            WHEN {trades_col} IS NOT NULL AND {trades_col} > 0
            THEN ({wins_col}::float / NULLIF({trades_col},0)) * 100.0
            ELSE NULL
        END"""

        # WHERE
        where = [ "simul_id = %s" ]
        params = [simul_id]

        if trades_min is not None:
            where.append(f"{trades_col} >= %s"); params.append(int(trades_min))
        if trades_max is not None:
            where.append(f"{trades_col} <= %s"); params.append(int(trades_max))

        if profit_min is not None:
            where.append(f"{profit_expr} >= %s"); params.append(float(profit_min))
        if profit_max is not None:
            where.append(f"{profit_expr} <= %s"); params.append(float(profit_max))

        if win_min is not None:
            where.append(f"({win_rate_expr}) >= %s"); params.append(float(win_min))
        if win_max is not None:
            where.append(f"({win_rate_expr}) <= %s"); params.append(float(win_max))

        where_sql = " AND ".join(where)

        # ORDER
        if sort_by == 'win_rate':
            order_sql = f"({win_rate_expr}) {sort_dir.upper()} NULLS LAST, {profit_expr} DESC NULLS LAST, {gid_col} ASC"
        else:
            order_sql = f"{profit_expr} {sort_dir.upper()} NULLS LAST, ({win_rate_expr}) DESC NULLS LAST, {gid_col} ASC"

        # COUNT
        with conn.cursor() as cur:
            sql_cnt = f"SELECT COUNT(*) FROM {qualify(RESULT_TABLE)} WHERE {where_sql}"
            cur.execute(sql_cnt, params)
            total = int(cur.fetchone()[0])

        # SELECT page
        offset = (page-1)*page_size
        select_cols = ", ".join(sel_min + [f"({win_rate_expr}) AS win_rate"])
        sql = f"""
            SELECT {select_cols}
            FROM {qualify(RESULT_TABLE)}
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT %s OFFSET %s
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params + [page_size, offset])
            rows = cur.fetchall()

    # Post-process: convert Decimal, build param_summary like legacy
    from decimal import Decimal as _D
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, _D):
                r[k] = float(v)
        # param_summary
        params_list = []
        for i in range(1, 37):
            nk, vk = f"p{i}_name", f"p{i}_val"
            name = r.get(nk, None)
            val  = r.get(vk, None)
            if name is not None and str(name).strip() != '' and val is not None:
                params_list.append(f"{name}={val}")
        r['param_summary'] = "; ".join(params_list) if params_list else None

    pages = (total + page_size - 1) // page_size if total else 0
    return {'items': rows, 'total': total, 'page': page, 'page_size': page_size, 'pages': pages}
# =============================
# Pages
# =============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cuda')
def cuda_entry():
    # 쿼리로 들어오는 경우 정규 경로로 리다이렉트
    sid = request.args.get('simul_id', type=int)
    if sid and sid > 0:
        return redirect(url_for('cuda_detail_page', simul_id=sid))
    return render_template('simul_list.html')

@app.route('/cuda/<int:simul_id>')
def cuda_detail_page(simul_id: int):
    return render_template('simul_result.html', simul_id=simul_id)

@app.route('/simul')
def simul_regist_page():
    return render_template('simul_regist.html')

# =============================
# APIs
# =============================
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

@app.route('/api/simul_list')
def api_simul_list():
    try:
        limit = int(request.args.get('limit', '200'))
        limit = max(1, min(limit, 2000))
        rows = fetch_simul_list(limit=limit)
        return jsonify({'items': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simul_detail')
def api_simul_detail():
    simul_id = request.args.get('simul_id', type=int)
    if simul_id is None or simul_id < 0:
        return jsonify({'error': 'simul_id is required (> 0)'}), 400
    # Back-compat: allow old 'limit' as page_size when no explicit page_size
    limit_legacy = request.args.get('limit', type=int)
    page        = request.args.get('page', type=int) or 1
    page_size   = request.args.get('page_size', type=int) or (limit_legacy or 200)

    # Range filters
    win_min     = request.args.get('win_min', type=float)
    win_max     = request.args.get('win_max', type=float)
    profit_min  = request.args.get('profit_min', type=float)
    profit_max  = request.args.get('profit_max', type=float)
    trades_min  = request.args.get('trades_min', type=int)
    trades_max  = request.args.get('trades_max', type=int)

    # Sorting
    sort_by     = (request.args.get('sort') or 'profit')
    sort_dir    = (request.args.get('dir')  or 'desc')

    try:
        data = fetch_simul_detail_paged(
            simul_id,
            page=page, page_size=page_size,
            win_min=win_min, win_max=win_max,
            profit_min=profit_min, profit_max=profit_max,
            trades_min=trades_min, trades_max=trades_max,
            sort_by=sort_by, sort_dir=sort_dir
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/apply_env_update', methods=['POST'])
def api_apply_env_update():
    """
    JSON 예:
    {
      "env_no": 5,
      "updates": {
        "s1m1_real_entry_avg_spread": 1.1,
        "s1m1_real_entry_profit": 1.5,
        "s1m1_clear_negative_avg_spread": 1.6,
        "is_s1m1_avg3": 0, "is_s1m1_avg10": 1, ...
      }
    }
    """
    if not APPLY_ENABLED:
        return jsonify({
            'ok': False,
            'error': 'Apply is disabled by server. Set ALPHA_ENABLE_APPLY=1 and restart.'
        }), 403

    data = request.get_json(silent=True) or {}
    env_no = data.get('env_no', None)
    updates = data.get('updates', None)

    if not isinstance(env_no, int) or env_no <= 0:
        return jsonify({'ok': False, 'error': 'env_no must be positive integer'}), 400
    if not isinstance(updates, dict) or not updates:
        return jsonify({'ok': False, 'error': 'updates must be a non-empty object'}), 400

    try:
        with get_pg_conn() as conn:
            cols = get_table_columns(conn, ENV_TABLE)  # 해당 테이블 실제 컬럼 목록
            # 컬럼명 검증 + 존재 검증
            bad_ident = [c for c in updates.keys() if not valid_ident.match(c)]
            if bad_ident:
                return jsonify({'ok': False, 'error': f'invalid column identifiers: {bad_ident}'}), 400

            unknown = [c for c in updates.keys() if c not in cols]
            if unknown:
                return jsonify({'ok': False, 'error': f'unknown columns: {unknown}'}), 400

            set_cols = list(updates.keys())
            if not set_cols:
                return jsonify({'ok': False, 'error': 'no valid columns to update'}), 400

            # SET col=%s, col=%s ...
            set_clause = ', '.join([f'{c} = %s' for c in set_cols])
            sql = f'UPDATE {qualify(ENV_TABLE)} SET {set_clause} WHERE env_no = %s'
            params = [updates[c] for c in set_cols] + [env_no]

            with conn.cursor() as cur:
                cur.execute(sql, params)
                updated = cur.rowcount
            conn.commit()

        return jsonify({'ok': True, 'updated_columns': set_cols, 'rowcount': updated})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/simulation', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_simulation():
    try:
        if request.method == 'GET':
            limit = int(request.args.get('limit', '1000'))
            limit = max(1, min(limit, 5000))
            rows = fetch_simul_list(limit=limit)
            return jsonify({'ok': True, 'items': rows})

        elif request.method == 'POST':
            # 등록 (simul_id 수동 지정)
            payload = request.get_json(silent=True) or request.form.to_dict()
            data = _clean_simul_payload(payload)
            if not data.get('simul_id'):
                return jsonify({'ok': False, 'error': 'simul_id is required'}), 400

            with get_pg_conn() as conn:
                cols_exist = get_table_columns(conn, SIMUL_TABLE)
                data = _filter_existing_cols(conn, SIMUL_TABLE, data)
                if not data:
                    return jsonify({'ok': False, 'error': 'no valid columns in payload'}), 400

                cols = list(data.keys())
                params = [data[c] for c in cols]
                placeholders = ['%s'] * len(cols)

                # reg_time 컬럼이 있다면 NOW()로 넣어줌
                if 'reg_time' in cols_exist:
                    cols.append('reg_time')
                    placeholders.append('NOW()')

                sql = f'INSERT INTO {qualify(SIMUL_TABLE)} ({", ".join(cols)}) VALUES ({", ".join(placeholders)}) RETURNING simul_id'
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    new_id = cur.fetchone()[0]
                conn.commit()

            return jsonify({'ok': True, 'simul_id': new_id})

        elif request.method == 'PUT':
            # 수정
            payload = request.get_json(silent=True) or {}
            simul_id = payload.get('simul_id', None)
            if not simul_id:
                return jsonify({'ok': False, 'error': 'simul_id is required'}), 400
            simul_id = int(simul_id)

            data = _clean_simul_payload(payload)
            # simul_id는 SET 대상에서 제외
            data.pop('simul_id', None)

            with get_pg_conn() as conn:
                data = _filter_existing_cols(conn, SIMUL_TABLE, data)
                if not data:
                    return jsonify({'ok': False, 'error': 'no valid columns in payload'}), 400
                set_clause = ', '.join([f'{k}=%s' for k in data.keys()])
                sql = f'UPDATE {qualify(SIMUL_TABLE)} SET {set_clause} WHERE simul_id=%s'
                with conn.cursor() as cur:
                    cur.execute(sql, list(data.values()) + [simul_id])
                    rc = cur.rowcount
                conn.commit()

            return jsonify({'ok': True, 'rowcount': rc})

        elif request.method == 'DELETE':
            # 삭제
            simul_id = request.args.get('simul_id', type=int)
            if not simul_id:
                data = request.get_json(silent=True) or {}
                simul_id = int(data.get('simul_id', 0))
            if not simul_id:
                return jsonify({'ok': False, 'error': 'simul_id is required'}), 400

            with get_pg_conn() as conn:
                sql = f'DELETE FROM {qualify(SIMUL_TABLE)} WHERE simul_id=%s'
                with conn.cursor() as cur:
                    cur.execute(sql, (simul_id,))
                    rc = cur.rowcount
                conn.commit()
            return jsonify({'ok': True, 'rowcount': rc})

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# =========[ Pages ]=========
@app.route('/paramset')
def paramset_page():
    # 템플릿 추가
    return render_template('param_regist.html')

# =========[ APIs ]=========
@app.route('/api/paramset/ids')
def api_paramset_ids():
    """set_id 목록(행 수 포함)"""
    try:
        with get_pg_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = f'''
                SELECT set_id, COUNT(*) AS row_count
                FROM {qualify(PARAM_TABLE)}
                GROUP BY set_id
                ORDER BY set_id ASC
            '''
            cur.execute(sql)
            rows = cur.fetchall()
        return jsonify({'ok': True, 'items': rows})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/paramset/rows')
def api_paramset_rows():
    """특정 set_id의 전체 파라메터 행과 총 경우의 수"""
    set_id = request.args.get('set_id', type=int)
    if set_id is None:
        return jsonify({'ok': False, 'error': 'set_id is required'}), 400
    try:
        rows, total_cases = _fetch_paramset_rows(set_id)
        return jsonify({'ok': True, 'items': rows, 'total_cases': total_cases})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/paramset/upsert', methods=['POST'])
def api_paramset_upsert():
    """
    Insert or Update 1 row
    JSON:
    {
      "set_id": 0,
      "param_no": 3,
      "param_name": "s1m1_clear_negative_avg_spread",
      "param_desc": "desc..",
      "data_type": "float" or "int",
      "data_min": 1.6,
      "data_max": 2.4,
      "data_step_size": 0.1,
      "is_active": 1
    }
    data_cnt는 서버에서 자동 계산/저장
    """
    payload = request.get_json(silent=True) or {}
    required = ['set_id', 'param_no', 'param_name', 'data_type']
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({'ok': False, 'error': f'missing: {missing}'}), 400

    set_id      = int(payload['set_id'])
    param_no    = int(payload['param_no'])
    param_name  = str(payload.get('param_name') or '').strip()
    param_desc  = str(payload.get('param_desc') or '').strip() or None
    data_type   = str(payload.get('data_type') or '').strip().lower()
    data_min    = payload.get('data_min', None)
    data_max    = payload.get('data_max', None)
    data_step   = payload.get('data_step_size', None)
    is_active   = int(payload.get('is_active', 1))

    if data_type not in ('float', 'int'):
        return jsonify({'ok': False, 'error': 'data_type must be "float" or "int"'}), 400

    # None 허용하지만, 셋이 다 들어오면 cnt 계산
    data_cnt = 0
    if data_min is not None and data_max is not None and data_step is not None:
        data_cnt = _calc_data_cnt(data_min, data_max, data_step)

    try:
        with get_pg_conn() as conn, conn.cursor() as cur:
            sql = f'''
                INSERT INTO {qualify(PARAM_TABLE)}
                    (set_id, param_no, param_name, param_desc, data_type,
                     data_min, data_max, data_step_size, data_cnt, is_active)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (set_id, param_no) DO UPDATE SET
                    param_name      = EXCLUDED.param_name,
                    param_desc      = EXCLUDED.param_desc,
                    data_type       = EXCLUDED.data_type,
                    data_min        = EXCLUDED.data_min,
                    data_max        = EXCLUDED.data_max,
                    data_step_size  = EXCLUDED.data_step_size,
                    data_cnt        = EXCLUDED.data_cnt,
                    is_active       = EXCLUDED.is_active
            '''
            cur.execute(sql, (
                set_id, param_no, param_name, param_desc, data_type,
                data_min, data_max, data_step, data_cnt, is_active
            ))
            conn.commit()
        rows, total_cases = _fetch_paramset_rows(set_id)
        return jsonify({'ok': True, 'data_cnt': data_cnt, 'total_cases': total_cases, 'items': rows})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/paramset/delete', methods=['DELETE'])
def api_paramset_delete():
    set_id   = request.args.get('set_id', type=int)
    param_no = request.args.get('param_no', type=int)
    if set_id is None or param_no is None:
        return jsonify({'ok': False, 'error': 'set_id and param_no are required'}), 400
    try:
        with get_pg_conn() as conn, conn.cursor() as cur:
            sql = f'DELETE FROM {qualify(PARAM_TABLE)} WHERE set_id=%s AND param_no=%s'
            cur.execute(sql, (set_id, param_no))
            rc = cur.rowcount
            conn.commit()
        rows, total_cases = _fetch_paramset_rows(set_id)
        return jsonify({'ok': True, 'rowcount': rc, 'items': rows, 'total_cases': total_cases})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500




# =============================
# Combined env update (main + sub-strategy) + simul ids tagging
# =============================

PERIODS = [3,5,7,10,15,20,25,30,35,40,45,50,55,60,70,80,120]
AVG_TARGETS = {
    "s1m1_entry_avg1","s1m1_entry_avg2",
    "s1m1_congest_avg1","s1m1_congest_avg2",
    "s1m1_congest2_avg1","s1m1_congest2_avg2",
    "s1m1_congest3_avg1","s1m1_congest3_avg2",
}

CONGEST_PREFIXES = ("s1m1_congest",)  # only sub-strategy congest* params overlay

def _fmt_num_py(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return '1' if v else '0'
    try:
        fv = float(v)
        if abs(fv - int(fv)) < 1e-12:
            return str(int(fv))
        s = f"{fv:.6f}"
        s = re.sub(r"\.0+$", "", s)
        s = re.sub(r"(\.\d*[1-9])0+$", r"\1", s)
        return s
    except Exception:
        return str(v)

def _apply_avg_slot_conversion_py(param_map: dict) -> dict:
    """
    Convert *_avg1 (slot index) and *_avg2 (distance) into real periods from PERIODS[],
    for keys listed in AVG_TARGETS. For entry_avg1/2 we only use them later to build flags.
    """
    out = dict(param_map)
    for key in list(out.keys()):
        if key in AVG_TARGETS and key.endswith('_avg1'):
            v1 = out.get(key, None)
            if v1 is None:
                continue
            try:
                idx1 = max(0, min(len(PERIODS)-1, int(float(v1))))
            except Exception:
                continue
            # Replace avg1 with actual period
            out[key] = PERIODS[idx1]
            # Handle paired avg2 for non-entry targets (congest etc.)
            key2 = key.replace('_avg1', '_avg2')
            if key.startswith('s1m1_entry_'):
                # do not overwrite _avg2 here; entry avg2 is the distance for flags later
                continue
            if key2 in out and out[key2] is not None:
                try:
                    dist = int(float(out[key2]))
                except Exception:
                    continue
                idx2 = max(0, min(len(PERIODS)-1, idx1 + dist))
                out[key2] = PERIODS[idx2]
    return out

def _row_to_param_map_py(row: dict) -> dict:
    m = {}
    # Collect p1_name/val .. p20_name/val
    for i in range(1, 37):
        nk = f'p{i}_name'
        vk = f'p{i}_val'
        name = row.get(nk, None)
        val  = row.get(vk, None)
        if name is not None and str(name).strip() != '' and val is not None:
            try:
                m[str(name).strip()] = float(val)
            except Exception:
                continue
    # fallback: if param_summary exists
    ps = row.get('param_summary')
    if ps and not m:
        try:
            for part in str(ps).split(';'):
                s = part.strip()
                if not s:
                    continue
                if '=' not in s:
                    continue
                k, v = s.split('=', 1)
                try:
                    m[k.strip()] = float(v.strip())
                except Exception:
                    pass
        except Exception:
            pass
    return m

def _fetch_simul_meta(simul_id: int):
    with get_pg_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        sql = f'''
            SELECT simul_id, simul_type, main_simul_id, main_simul_gid
            FROM {qualify(SIMUL_TABLE)}
            WHERE simul_id = %s
        '''
        cur.execute(sql, (simul_id,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f'simul_id {simul_id} not found in {DB_SCHEMA}.{SIMUL_TABLE}')
        return row

def _resolve_gid_col(conn):
    cols = get_table_columns(conn, RESULT_TABLE)
    gid_col, _wins, _trades, _profit_expr = _resolve_result_columns(cols)
    return gid_col, cols

def _fetch_result_row(simul_id: int, gid: int):
    with get_pg_conn() as conn:
        gid_col, cols = _resolve_gid_col(conn)
        sel = [f'{gid_col} AS gid']
        for i in range(1, 37):
            nk, vk = f'p{i}_name', f'p{i}_val'
            if nk in cols: sel.append(nk)
            if vk in cols: sel.append(vk)
        # optional param_summary
        if 'param_summary' in cols:
            sel.append('param_summary')
        sql = f'''
            SELECT {', '.join(sel)}
            FROM {qualify(RESULT_TABLE)}
            WHERE simul_id = %s AND {gid_col} = %s
            LIMIT 1
        '''
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (simul_id, gid))
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f'Result row not found for simul_id={simul_id}, gid={gid}')
            return row

def _build_updates_from_maps(env_no: int, main_map: dict, sub_map: dict):
    """
    Merge main and sub parameter maps; see rules in code.
    """
    main_conv = _apply_avg_slot_conversion_py(main_map or {})
    sub_conv  = _apply_avg_slot_conversion_py(sub_map or {})

    updates = {}
    for k, v in main_conv.items():
        if k in ('s1m1_entry_avg1', 's1m1_entry_avg2'):
            continue
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        updates[k] = fv

    for k, v in sub_conv.items():
        if not any(k.startswith(pref) for pref in CONGEST_PREFIXES):
            continue
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        updates[k] = fv

    slot1_raw = main_map.get('s1m1_entry_avg1', None)
    dist_raw  = main_map.get('s1m1_entry_avg2', None)
    picked = set()
    if slot1_raw is not None:
        try:
            idx1 = max(0, min(len(PERIODS)-1, int(float(slot1_raw))))
            picked.add(PERIODS[idx1])
            if dist_raw is not None:
                idx2 = max(0, min(len(PERIODS)-1, idx1 + int(float(dist_raw))))
                picked.add(PERIODS[idx2])
        except Exception:
            pass
    if picked:
        for p in PERIODS:
            updates[f'is_s1m1_avg{p}'] = 1 if p in picked else 0

    set_parts = [f"{k}={_fmt_num_py(v)}" for k, v in updates.items()]
    sql = f"UPDATE {qualify(ENV_TABLE)} SET " + ", ".join(set_parts) + f" WHERE env_no={int(env_no)};"
    return sql, updates

@app.route('/api/combined_env_update')
def api_combined_env_update():
    """Build combined UPDATE SQL (and updates JSON) for a clicked row."""
    sub_simul_id = request.args.get('sub_simul_id', type=int)
    gid          = request.args.get('gid', type=int)
    env_no       = request.args.get('env_no', type=int)
    if not sub_simul_id or not gid or not env_no:
        return jsonify({'ok': False, 'error': 'sub_simul_id, gid, env_no are required'}), 400

    try:
        meta = _fetch_simul_meta(sub_simul_id)
        simul_type   = int(meta.get('simul_type') or 0)
        main_simul_id = meta.get('main_simul_id')
        main_simul_gid = meta.get('main_simul_gid')

        if simul_type == 0:
            row = _fetch_result_row(sub_simul_id, gid)
            main_map = _row_to_param_map_py(row)
            sql, updates = _build_updates_from_maps(env_no, main_map, {})
        else:
            if not main_simul_id or not main_simul_gid:
                return jsonify({'ok': False, 'error': 'Missing main_simul_id/main_simul_gid in simul table'}), 400
            row_sub  = _fetch_result_row(sub_simul_id, gid)
            row_main = _fetch_result_row(int(main_simul_id), int(main_simul_gid))
            main_map = _row_to_param_map_py(row_main)
            sub_map  = _row_to_param_map_py(row_sub)
            sql, updates = _build_updates_from_maps(env_no, main_map, sub_map)

        # Attach simul ids/gids into updates
        if simul_type == 0:
            main_id = int(sub_simul_id)
            main_gid = int(gid)
            sub_id = 0
            sub_gid = 0
        else:
            main_id = int(main_simul_id)
            main_gid = int(main_simul_gid)
            sub_id = int(sub_simul_id)
            sub_gid = int(gid)
        updates['simul_id'] = main_id
        updates['simul_gid'] = main_gid
        updates['sub_simul_id'] = sub_id
        updates['sub_simul_gid'] = sub_gid
        # Rebuild SQL to include the four new columns as well
        set_parts = [f"{k}={_fmt_num_py(v)}" for k, v in updates.items()]
        sql = f"UPDATE {qualify(ENV_TABLE)} SET " + ", ".join(set_parts) + f" WHERE env_no={int(env_no)};"


        return jsonify({
            'ok': True,
            'sql': sql,
            'updates': updates,
            'simul_type': simul_type,
            'main_simul_id': main_simul_id,
            'main_simul_gid': main_simul_gid,
            'main_gid': (int(main_simul_gid) if main_simul_gid else None),
            'sub_simul_id': int(sub_simul_id),
            'sub_gid': int(gid)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# === ADD: New API endpoints ===
@app.route('/api/io')
def api_io():
    disk, net = get_disk_net_rates()
    return jsonify({'disk': disk, 'net': net})

@app.route('/api/net')
def api_net():
    _, net = get_disk_net_rates()
    return jsonify(net)

@app.route('/api/gpu_extra')
def api_gpu_extra():
    return jsonify(get_gpu_extra())

@app.route('/api/fans')
def api_fans():
    return jsonify(get_cpu_fans())

@app.route('/api/processes')
def api_processes():
    return jsonify(get_top_processes())

@app.route('/api/gpu_processes')
def api_gpu_processes():
    return jsonify(get_gpu_processes())




# =============================
# Main
# =============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

