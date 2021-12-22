import json
import matplotlib.pyplot as plt
import numpy as np

color_set = {
        'Amaranth': np.array([0.9, 0.17, 0.31]),  # main algo
        'Amber': np.array([1.0, 0.49, 0.0]),  # main baseline
        'Bleu de France': np.array([0.19, 0.55, 0.91]),
        'Electric violet': np.array([0.56, 0.0, 1.0]),
        'Dark sea green': 'forestgreen',
        'Dark electric blue': 'deeppink',
        'Dark gray': np.array([0.66, 0.66, 0.66]),
        'Arsenic': np.array([0.23, 0.27, 0.29]),
    }

color_list = []
for key,value in color_set.items():
    color_list.append(value)

def legend():
    plt.figlegend(loc='upper right', prop={'size': 26.0}, frameon=True, ncol=1)
    plt.tight_layout()

def smooth(data, smooth_range):
	# print('hhhhhhh', type(data), len(data))
	new_data = np.zeros_like(data)
	for i in range(0, data.shape[-1]):
		if i < smooth_range:
			new_data[:, i] = 1. * np.sum(data[:, :i + 1], axis=1) / (i + 1)
		else:
			new_data[:, i] = 1. * np.sum(data[:, i - smooth_range + 1:i + 1], axis=1) / smooth_range

	return new_data

def read_data(paths=[],smoothed=30):
	final_datas = {'x':[],'sr':[],'srs':[]}
	min_length = 100000000000
	for p in paths:
		datas = []
		with open(p,'r') as f:
			for l in f.readlines():
				datas.append(json.loads(l))
		x = np.zeros((len(datas)))
		sr = np.zeros((len(datas)))
		# srs = np.zeros((25, len(datas)))
		for num, d in enumerate(datas):
			x[num] = d['episode']
			sr[num] = d['success']
			# for j in range(25):
			# 	srs[j][num] = d['success_env_index_' + '%d' % j]
		sr = smooth(sr[None], smoothed)[0]
		# srs = smooth(srs, smoothed)
		final_datas['x'].append(x[None])
		final_datas['sr'].append(sr[None])
		if x.shape[0]<min_length:
			min_length = x.shape[0]
	for i in range(len(final_datas['x'])):
		final_datas['x'][i] = final_datas['x'][i][:,:min_length]
		final_datas['sr'][i] = final_datas['sr'][i][:, :min_length]
	final_datas['x'] = np.concatenate(final_datas['x'],0)
	final_datas['sr'] = np.concatenate(final_datas['sr'], 0)
	return final_datas, min_length


pcgrad_data, pcgrad_length = read_data(['/data2/zj/mtrl/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_2/train.log','/data2/zj/mtrl/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_6/train.log','/data2/zj/mtrl/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_9/train.log'],30)

mine_data, mine_length = read_data(['/data2/zj/mtrl/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_5/train.log','/data2/zj/mtrl/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_7/train.log','/data2/zj/mtrl/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_8/train.log'],30)

# with open('/data2/zj/mtrl/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_6/train.log','r') as f:
# 	for l in f.readlines():
# 		datas.append(json.loads(l))
#
# x=np.zeros((len(datas)))
# sr = np.zeros((len(datas)))
# srs = np.zeros((25,len(datas)))
# for num,d in enumerate(datas):
# 	x[num] = d['episode']
# 	sr[num] = d['success']
# 	for j in range(25):
# 		srs[j][num] = d['success_env_index_'+'%d'%j]
# sr = smooth(sr[None],30)[0]
# # print(datas)

plt.figure(figsize=(8.5,6))
plt.style.use('seaborn-whitegrid')
plt.rc('font', family='Times New Roman')
# matplotlib.rcParams['text.usetex'] = True
plt.clf()
ax = plt.gca()
for d,c,n in zip([pcgrad_data,mine_data],[1,0],['PcGrad','Mine']):
	mean = np.mean(d['sr'],0)
	std = np.std(d['sr'],0)
	ax.fill_between(d['x'][0], mean - std, mean + std, alpha=0.1, color=color_list[c],
					linewidth=0)
	ax.plot(d['x'][0], mean, color=color_list[c],
			label=n, linewidth=3)

legend()
plt.show()
# plt.figure()
# for i in range(25):
# 	plt.plot(x,srs[i,:])
# plt.show()
# print(np.mean(srs[:,-5:],axis=-1))