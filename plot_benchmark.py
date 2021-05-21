# Plot graphs showcasing execution time and speedup

import pandas as pd
import matplotlib.pyplot as plt

# Process input
df = pd.read_csv("times.csv")
df['speedup'] = df['CPU Time(ms)'] / df['GPU Time(ms)']

# Create plot arrays
x = df.iloc[:,0] / 10000
y1 = df['CPU Time(ms)']
y2 = df['GPU Time(ms)']
y3 = df['speedup']

print('Generating CPU vs GPU plot')

# Plot CPU vs GPU times
fig = plt.figure()
plt.title('Variation of computation time with No of cells')
plt.semilogy(x,y1,color='red',label='CPU')
plt.semilogy(x,y2,color='blue',label='GPU')
plt.xlabel('No of cells (in ten thousands)')
plt.ylabel('Compute Time(ms)')
plt.legend()
plt.savefig('assets/ComputeTime.png',format='png',dpi=1200)

print('Generating speedup plot')

# Plot speedup
fig = plt.figure()
plt.title('GPU speedup vs No of Cells')
plt.semilogx(x,y3,color='green')
plt.xlabel('No of cells (in ten thousands)')
plt.ylabel('Speedup')
plt.savefig('assets/Speedup.png',format='png',dpi=1200)