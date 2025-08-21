# 접속 정보
HOST="192.168.0.200"
USER="woo"
DB="stock"
SCHEMA="alpha"

# 실행
psql -h "$HOST" -U "$USER" -d "$DB" -c "
INSERT INTO ${SCHEMA}.tb_wft_min AS t
SELECT * FROM alpha_src.tb_wft_min AS s
WHERE s.dt > '20250801000000'
ON CONFLICT (item_name, unit, dt) DO NOTHING;
"
