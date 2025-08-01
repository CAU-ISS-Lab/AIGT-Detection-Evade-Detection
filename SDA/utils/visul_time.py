import matplotlib.pyplot as plt

time=[2749.5991,2514.5991,2907]
label=['Paraphrase','HMGC','SDA(ours)']
colors=['#','#','#']

plt.figure(figsize=(8,6))
plt.bar(label,time,color=colors)
plt.set_ylabel('Execution Time (s)',fontsize=14)
plt.show()
plt.savefig("visul/time.png")
