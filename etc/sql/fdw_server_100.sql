
-- 타겟(200)의 stock DB에서
CREATE EXTENSION IF NOT EXISTS postgres_fdw;

CREATE SERVER src_srv
  FOREIGN DATA WRAPPER postgres_fdw
  OPTIONS (host '192.168.0.100', dbname 'stock', port '5432');

CREATE USER MAPPING FOR woo
  SERVER src_srv
  OPTIONS (user 'woo', password '소스DB_woo_비번');

-- 스키마 충돌 방지용으로 별도 스키마를 하나 준비
CREATE SCHEMA IF NOT EXISTS alpha_src;

-- 소스 alpha 스키마에서 해당 테이블만 가져오기
IMPORT FOREIGN SCHEMA alpha
  LIMIT TO (tb_wft_min)
  FROM SERVER src_srv
  INTO alpha_src;

CREATE USER MAPPING FOR woo
  SERVER src_srv
  OPTIONS (user 'woo', password 'db4woo');


----------------------------------------------------
--INSERT INTO alpha.tb_wft_min AS t
--SELECT * FROM alpha_src.tb_wft_min AS s
--WHERE s.dt > '20250801000000'
--ON CONFLICT (item_name, unit, dt) DO NOTHING;

