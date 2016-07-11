#!/usr/bin/env bash
sudo perf stat -a -e r500fa,r600f4,r200fd,r3e054,r300f0,r4c042,LLC-store-misses,r400f6,r100f6,r200f6,r62c04c,r600f4 -I 500 -o /home/PBDST/perfout sleep 100000
