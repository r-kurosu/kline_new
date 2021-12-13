#!/bin/sh

/usr/bin/python3 /home/takeda/KLINE/model2-hybrid.py book/exp.csv > model2-hybrid-log/exp.out
/usr/bin/python3 /home/takeda/KLINE/model2-hybrid.py book/exp_2_5.csv > model2-hybrid-log/exp_2_5.out
/usr/bin/python3 /home/takeda/KLINE/model2-hybrid.py book/exp_4_3.csv > model2-hybrid-log/exp_4_3.out
/usr/bin/python3 /home/takeda/KLINE/model2-hybrid.py book/exp_4_3.csv > model2-hybrid-log/level1-1.out
