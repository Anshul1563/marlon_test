import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


csvs = ["QL_4x4_E1.csv","QL_4x4_E2.csv"]
models = ["DQN", "A2C"]


datas = []

for c in csvs :
    datas.append(pd.read_csv(c))

steps = datas[0]["step"]

queues = []
waiting_times = []

for d in datas :
    queues.append(d['system_total_stopped'])
    waiting_times.append(d['system_total_waiting_time'])


plt.figure(1)
i = 0
for q in queues :
    plt.plot(steps, q, label = models[i])
    i +=1
plt.xlabel('Steps')
plt.ylabel('Queue Length')
plt.title('Comparision of 2 alogrithms')
plt.legend()



plt.figure(2)
i = 0
for w in waiting_times :
    plt.plot(steps, w, label = models[i])
    i +=1
plt.xlabel('Steps')
plt.ylabel('Total Waiting Time')
plt.title('Comparision of 2 alogrithms')
plt.legend()


with PdfPages('plots.pdf') as pdf:
    pdf.savefig(1)
    pdf.savefig(2)

plt.show()

