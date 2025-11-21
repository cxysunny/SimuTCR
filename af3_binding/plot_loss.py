import matplotlib.pyplot as plt
tr_loss = []
tr_step = []
model_name = "seq_plus_structure_6.16" 

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

tr_step = [i for i in range(0, 10, 1)]
tr_loss = [0.6146, 0.6482, 0.6699, 0.6950, 0.7025, 0.6899, 0.7234, 0.7391, 0.7375, 0.7595]
ori_loss = [0.5355, 0.5721, 0.5919, 0.6278, 0.5858, 0.6560, 0.6262, 0.6346, 0.6209, 0.5912]
seq_loss = [0.6122, 0.6698, 0.6442,0.6699, 0.6998,0.7003,0.6972,0.7077,0.7122,0.7107 ]
plt.figure(figsize=(8,6))
plt.plot(tr_step,tr_loss,'',label="full",color='red')
plt.plot(tr_step,ori_loss,'',label="stru only",color='blue')
plt.plot(tr_step,seq_loss,'',label="seq only",color='green')
plt.title('Training test AUC')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Test Binding AUC')
plt.savefig("/home/xycui/project/af3_binding/af3_binding/photo/"+'/train_auc.svg',format='svg')

tr_step = [i for i in range(0, 10, 1)]
tr_loss = [0.3731, 0.4241, 0.4433, 0.4793, 0.4775, 0.4757, 0.5153, 0.5398, 0.5255, 0.5400]
ori_loss = [0.3449, 0.3787, 0.3846, 0.4249, 0.4303, 0.4889, 0.4290, 0.4303, 0.4710, 0.4102]
seq_loss = [0.3702, 0.4528, 0.4056,0.4367, 0.4884,0.4600,0.4585,0.4876,0.4950,0.4930 ]
plt.figure(figsize=(8,6))
plt.plot(tr_step,tr_loss,'',label="full",color='red')
plt.plot(tr_step,ori_loss,'',label="stru only",color='blue')
plt.plot(tr_step,seq_loss,'',label="seq only",color='green')
plt.title('Training test AUPR')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Test Binding AUPR')
plt.savefig("/home/xycui/project/af3_binding/af3_binding/photo/"+'/train_aupr.svg',format='svg')

# tr_step = [i for i in range(0, 45, 5)]
# tr_loss = [0.5994, 0.6089, 0.6311, 0.5834, 0.6312, 0.5807, 0.5809, 0.5832, 0.5881]

# plt.figure(figsize=(8,6))
# plt.plot(tr_step,tr_loss,'',label="version-now")
# plt.title('ood auc')
# plt.legend(loc='upper right')
# plt.xlabel('epoch ')
# plt.ylabel('ood auc')
# plt.savefig("/home/xycui/project/af3_binding/results/"+ model_name+'/ood_auc.jpg')
