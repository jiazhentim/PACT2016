#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Usage: start-slave.sh &lt;worker#&gt; 
#   where  is like "spark://localhost:7077"

sbin="`dirname "$0"`"
sbin="`cd "$sbin"; pwd`"

. conf/spark-env.sh   # load SPARK_WORKER_INSTANCES/SPARK_WORKER_CORES variables
MACHINE_THREADS=192  #120 for x3850

CPU_START=$(( ($1-1) * MACHINE_THREADS/SPARK_WORKER_INSTANCES ))
DELTA=$(( MACHINE_THREADS/SPARK_WORKER_INSTANCES/SPARK_WORKER_CORES ))
CPU=$CPU_START
for N in $(seq 2 $SPARK_WORKER_CORES)
do
	    CPU="$CPU,$(( (CPU_START + (N-1)*DELTA) % MACHINE_THREADS ))"
    done


    echo numactl --physcpubind=$CPU --localalloc \
    "$sbin"/spark-daemon.sh start org.apache.spark.deploy.worker.Worker "$@"
    numactl --physcpubind=$CPU --localalloc \
    "$sbin"/spark-daemon.sh start org.apache.spark.deploy.worker.Worker "$@"

    # echo "$sbin"/spark-daemon.sh start org.apache.spark.deploy.worker.Worker "$@"
    #"$sbin"/spark-daemon.sh start org.apache.spark.deploy.worker.Worker "$@"

