## Script queries GoL.cu and graphs exection time for both the CPU and the GPU

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

print("BEGINNING BENCHMARK")

## Compile target using make
proc = subprocess.Popen(["make", "test"])
proc.wait()

## Define max number of generations
maxgen = 10000

## Define starting set of rows and columns
(rows, columns) = (10, 10)

## Define lists for appending results to
cells = []
cpu_times = []
gpu_times = []

print("ESTIMATING EXECUTION TIME")

## Start calculation for configurations
for i in range(100):

    proc = subprocess.Popen(["./bencher","-r",str(rows),"-c",str(columns),"-g",str(maxgen)])
    proc.wait()

    # Read O/P
    fp = open('times.txt','r')
    arr = fp.readlines()

    cpu_times.append(float(arr[0]))
    gpu_times.append(float(arr[1]))
    cells.append(rows*columns)
    
    fp.close()

    rows += 10
    columns += 10

## Plot results
print("GENERATING PLOTS")

fig = plt.figure(figsize=(10,10))
plt.plot(cpu_times,cells,"r-",label="CPU")
plt.plot(gpu_times,cells,'b-',label="GPU")
plt.legend()
plt.xlabel("Number of cells")
yname = "Exection time (" + str(maxgen) + " generations)"
plt.ylabel(yname)
figname = "Benchmark_plot_gol"
plt.savefig(figname,dpi=300)

## Clean targets
proc = subprocess.Popen(["make", "clean"])
proc.wait()

print("END")
