import subprocess
import matplotlib.pyplot as plt


subprocess.run(["mpicc","trapIntegral-hw.c", "-o", "out"], check=True)
processor_count = [1,2,3,4]


speedups = []
efficiencies = []
processors = []

for p in processor_count:

    result = subprocess.run(
        ["mpirun", "-np", str(p), "./out"],
        capture_output=True, text=True
    )

    output_lines = result.stdout.splitlines()
    #print(output_lines) #debugging puposes
    sequential_time = float(output_lines[4].split(":")[1].strip())
    parallel_time = float(output_lines[8].split(":")[1].strip())

    speedup = sequential_time/parallel_time
    efficiency = (speedup/p)*100
    speedups.append(speedup)
    processors.append(p)
    efficiencies.append(efficiency)


#for speedup
plt.figure()
plt.plot(processors, speedups, marker='o')
plt.title('Speed-up Factor as a function of Number of Processors')
plt.xlabel('Number of Processors')
plt.ylabel('Speed-up Factor')
plt.grid(True)
plt.show()

#for effeciency
plt.figure()
plt.bar(processors,efficiencies, edgecolor="black")
plt.title("histogram showing efficiency of program")
plt.xlabel("processor")
plt.ylabel("percent")
plt.grid(True)
plt.show()
