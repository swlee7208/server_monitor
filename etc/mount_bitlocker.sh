#!/usr/bin/env bash
sudo dislocker -V /dev/sda1 --user-password=11111111 -- /mnt/bitlocker
sleep 1
sudo mount -t exfat -o loop,uid=1000,gid=1000,umask=000 \
  /mnt/bitlocker/dislocker-file /mnt/bitlocker-unlocked
