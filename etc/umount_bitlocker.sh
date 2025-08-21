#!/usr/bin/env bash
sudo umount /mnt/bitlocker-unlocked 2>/dev/null || true
sudo umount /mnt/bitlocker 2>/dev/null || sudo fusermount3 -u /mnt/bitlocker 2>/dev/null || true

