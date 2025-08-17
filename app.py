from flask import Flask, render_template, jsonify, request, redirect, url_for
import psutil
import subprocess
import os
import re
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

ENV_TABLE = os.getenv('ALPHA_ENV_TABLE', 'alpha_env')
APPLY_ENABLED = os.getenv('ALPHA_ENABLE_APPLY', '1')

valid_ident = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')  # 컬럼명 검증용


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
    'simul_id', 'simul_name', 'simul_type',
    'reg_time', 'simul_from_date', 'simul_to_date',
    'param_set_id',              # NEW
    'is_active',                 # NEW
    'status'
]

# 쓰기(등록/수정) 허용 컬럼 (요청한 입력 항목)
SIMUL_WRITE_FIELDS = [
    'simul_id', 'simul_name', 'simul_desc', 'is_active',
    'simul_from_date', 'simul_to_date', 'simul_candle',
    'simul_type', 'main_simul_id', 'main_top_rank', 'param_set_id',
    'reason'
]

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
        if k in ('simul_id', 'simul_candle', 'simul_type',
                 'main_simul_id', 'main_top_rank', 'param_set_id', 'is_active'):
            data[k] = None if v is None or str(v).strip() == '' else int(v)
        elif k in ('simul_from_date', 'simul_to_date'):
            data[k] = _parse_ts(v)
        elif k in ('simul_name', 'simul_desc', 'reason'):
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
        if 'coretemp' in temps:
            return temps['coretemp'][0].current
        elif 'cpu-thermal' in temps:
            return temps['cpu-thermal'][0].current
    except Exception:
        return None
    return None

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

        sql = f'''
            SELECT {select_list}
            FROM {qualify(SIMUL_TABLE)}
            ORDER BY
                {order_prefix}
                simul_id DESC
            LIMIT %s
        '''
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

    # 날짜/시간 문자열화
    for r in rows:
        for k in ('simul_from_date', 'simul_to_date', 'end_time', 'reg_time'):
            if k in r and r[k] is not None:
                if isinstance(r[k], datetime):
                    r[k] = r[k].strftime('%Y-%m-%d %H:%M:%S')
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
        for i in range(1, 21):
            nk, vk = f'p{i}_name', f'p{i}_val'
            if nk in cols: select_opt.append(nk)
            if vk in cols: select_opt.append(vk)

        select_list = ', '.join(sel_min + select_opt) if select_opt else ', '.join(sel_min)

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
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (simul_id, limit))
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
        for i in range(1, 21):
            nk, vk = f'p{i}_name', f'p{i}_val'
            name = r.get(nk, None)
            val  = r.get(vk, None)
            if name is not None and str(name).strip() != '' and val is not None:
                params.append(f"{name}={val}")
        r['param_summary'] = "; ".join(params) if params else None

    return rows

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
    return render_template('cuda.html')

@app.route('/cuda/<int:simul_id>')
def cuda_detail_page(simul_id: int):
    return render_template('simul_detail.html', simul_id=simul_id)

@app.route('/simul')
def simul_regist_page():
    return render_template('simul_regist.html')

# (선택) 파라메터셋 등록 페이지도 쓸 경우
@app.route('/paramset')
def paramset_page():
    # templates/paramset.html 이 있다면 아래로 교체:
    # return render_template('paramset.html')
    return '<html><body><nav><a href="/">Server Monitor</a> | <a href="/cuda">Alpha Cuda Simul</a> | <a href="/simul">Simulation Regist</a> | <a href="/paramset">Parameta Set Regist</a></nav><div style="padding:16px"><h3>Parameta Set Regist (WIP)</h3></div></body></html>'


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
    limit = request.args.get('limit', default=500, type=int)
    try:
        limit = max(1, min(limit, 5000))
        rows = fetch_simul_detail(simul_id, limit=limit)
        return jsonify({'items': rows})
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



# =============================
# Main
# =============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

