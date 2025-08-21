#!/bin/bash

# 접속 정보
HOST="192.168.0.200"
USER="woo"
DB="stock"

# 백업 파일 경로 (날짜 붙여서 관리)
BACKUP_DIR="/mnt/bitlocker-unlocked/Backup/Alpha/SimulDB"
mkdir -p "$BACKUP_DIR"
BACKUP_FILE="${BACKUP_DIR}/${DB}_$(date +%Y%m%d_%H%M%S).sql"

# pg_dump 실행
pg_dump -h "$HOST" -U "$USER" -d "$DB" -F c -b -v -f "$BACKUP_FILE"

# 결과 메시지
if [ $? -eq 0 ]; then
  echo "✅ Backup completed: $BACKUP_FILE"
else
  echo "❌ Backup failed!"
  exit 1
fi

