#! /usr/bin/env python
import sys
import read_data
smt=sys.argv[1]
SMT =read_data.LR_predict('/home/PBDST/test_lable','/home/PBDST/perfout',smt)
#SMT =read_data.KNN_predict('/home/PBDST/test_lable','/home/PBDST/perfout',smt)

if SMT==0:
    print 0
else:
    print SMT-int(smt)
