import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import copy
from torch import optim
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.animation import FuncAnimation
import os
from sklearn.preprocessing import StandardScaler


import os
weight_dir = 'weights'
os.makedirs(weight_dir, exist_ok=True)

def combined_reward(rank, time_difference):
	if time_difference == -1:
		in_top_3_reward = 1  # time_differenceが-1の場合、最高の報酬を与える
	else:
		in_top_3_reward = 1 / (time_difference + 1) if rank <= 3 else 0
	penalty = -time_difference if rank > 3 else 0
	return in_top_3_reward + penalty


class Environment:
	def __init__(self, table):
		self.table = table
		self.race_id_list = table.race_group_id.unique()


	def get_state(self, race_id):

		target_race = self.table[self.table['race_group_id'] == race_id]
		target_race = target_race.copy()

		state = target_race[["prob_fukushou_hit_1",'ave_value','timeshisu_five_ave_ss','kishu_chakujyun_median_ss','zensou_chakusa_ss','nisou_chakusa_ss','onaji_kishu_median_ss','onaji_kyori_median_ss','timeshisu_max_ss','zensou_level_ss','nisou_level_ss','onaji_baba_median_ss','timeshisu_course_ss','timeshisu_kyori_ss','nige_group_pct','senkou_group_pct','sashi_group_pct','oikomi_group_pct','zensou_th_kyakushitsu_先','zensou_th_kyakushitsu_差','zensou_th_kyakushitsu_追','zensou_th_kyakushitsu_逃']].values
		# state = target_race[["ave_value","prediction_label", "prediction_score","zensou_th_kyakushitsu_codes","timeshisu_five_ave_ss","kishu_chakujyun_median_ss","zensou_chakusa_ss","nisou_chakusa_ss","onaji_kishu_median_ss","onaji_kyori_median_ss","timeshisu_max_ss","zensou_level_ss","nisou_level_ss","onaji_baba_median_ss","timeshisu_course_ss","timeshisu_kyori_ss","place_codes","race_type_codes"]].values
		return state


	def bet_ninki(self, race_id, horse_index):
		target_race = self.table[(self.table['race_group_id'] == race_id)]
		zensou_ninki_ss = target_race.iloc[horse_index]['zensou_ninki_ss'].astype(float)
		fukushou_hit = target_race.iloc[horse_index]['fukushou_hit'].astype(int)

		bet_money = 1 #100円購入

		# 馬券が当たった場合
		if fukushou_hit == 1:
			# 前走人気がなかった馬が3着以内に来たときほど報酬を大きくする
			reward = bet_money * (1 - zensou_ninki_ss) 
		# 馬券が外れた場合
		else:
			reward = -bet_money

		return reward, bet_money


	# Environment クラスの bet_wide 関数を新しい報酬に対応させる
	def bet_fukushou(self, race_id, horse_index):
		target_race = self.table[(self.table['race_group_id'] == race_id)]
		num_horses_in_race = len(target_race)  # 追加: レースごとの馬の数
		fuku_ozz = target_race.iloc[horse_index]['fuku_ozz'].astype(float)
		fukushou_hit = target_race.iloc[horse_index]['fukushou_hit'].astype(int)

		# fuku_ozz が 0 の場合、報酬とベット金額を 0 に設定
		if fuku_ozz == 0:
			reward = 0
			bet_money = 0
		else:
			bet_money = 1 #100円購入
			# bet_money = 1 / fuku_ozz

			# 馬券が当たった場合
			if fukushou_hit == 1:
				reward = fuku_ozz * bet_money
			# 馬券が外れた場合
			else:
				reward = -bet_money

		return reward, bet_money

	def bet_chakusa(self, race_id, horse_index):
		target_race = self.table[(self.table['race_group_id'] == race_id)]
		rank = target_race.iloc[horse_index]['chakujyun'].astype(int)
		time_difference = target_race.iloc[horse_index]['chakusa_time'].astype(float)

		external_reward = combined_reward(rank, time_difference)

		bet_money = 1
		return external_reward, bet_money


	# 新しい rank_horses 関数
	def rank_horses(self, race_id, predicted_rank):
		race_data = self.table[self.table['race_group_id'] == race_id]
		sorted_horses = race_data.iloc[predicted_rank.argsort()[::-1]]
		return sorted_horses

# 以下の部分は同じままで動作します。
class ReplayBuffer:
	def __init__(self,memory_size):
		self.memory_size = memory_size
		self.memory = deque([],maxlen = memory_size)

	def append(self,transition):
		self.memory.append(transition)

	def sample(self,batch_size):
		batch_indexes = np.random.randint(0,len(self.memory),size=batch_size)

		#予測スコアと単勝オッズの組み合わせ
		states = np.array([self.memory[index]['state'] for index in batch_indexes], dtype=object)
		# states = np.array([self.memory[index]['state'] for index in batch_indexes])

		# もらった報酬
		rewards = np.array([self.memory[index]['reward'] for index in batch_indexes], dtype=object)
		# rewards = np.array([self.memory[index]['reward'] for index in batch_indexes])

		#どの馬券にかけるか
		actions = np.array([self.memory[index]['action'] for index in batch_indexes], dtype=object)

		 # ベット金額
		bet_money = np.array([self.memory[index]['bet_money'] for index in batch_indexes], dtype=object)

		# next_states を追加
		# next_states = np.array([self.memory[index]['next_state'] for index in batch_indexes], dtype=object)

		# return {'states':states,"rewards":rewards,"actions":actions,"bet_money":bet_money, "next_states": next_states}
		return {'states':states,"rewards":rewards,"actions":actions,"bet_money":bet_money}


import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
	def __init__(self, num_state, dropout_rate=0.5):  # dropout_rateを引数に追加
		super(QNetwork, self).__init__()
		self.num_state = num_state
		self.fc1 = nn.Linear(num_state, 256)
		self.ln1 = nn.LayerNorm(256)
		self.dropout1 = nn.Dropout(dropout_rate)  # Dropoutレイヤを追加
		self.fc2 = nn.Linear(256, 512)
		self.ln2 = nn.LayerNorm(512)
		self.dropout2 = nn.Dropout(dropout_rate)  # Dropoutレイヤを追加
		self.fc3 = nn.Linear(512, 256)
		self.ln3 = nn.LayerNorm(256)
		self.dropout3 = nn.Dropout(dropout_rate)  # Dropoutレイヤを追加

		# 状態価値V(s)の層
		self.fc_value = nn.Linear(256, 1)

		# アクションアドバンテージA(s,a)の層
		self.fc_advantage = nn.Linear(256, 16)

	def forward(self, x, mask=None):
		x = self.fc1(x)
		x = self.ln1(x)
		x = self.dropout1(x)  # Dropoutを適用
		x = F.relu(x)
		x = self.fc2(x)
		x = self.ln2(x)
		x = self.dropout2(x)  # Dropoutを適用
		x = F.relu(x)
		x = self.fc3(x)
		x = self.ln3(x)
		x = self.dropout3(x)  # Dropoutを適用
		x = F.relu(x)

		# 状態価値V(s)を計算
		value = self.fc_value(x)

		# アクションアドバンテージA(s,a)を計算
		advantage = self.fc_advantage(x)

		# Q値を計算
		q = value + advantage - advantage.mean(dim=1, keepdim=True)

		if mask is not None:
			q = q.masked_fill(mask == 0, -1e9)
		return q


class DqnAgent:
	def __init__(self, num_state, lr=0.0001, batch_size=32, memory_size=50000,weight_decay=1e-5,dropout_rate=0.5):
		self.num_state = num_state
		self.batch_size = batch_size  # Q関数の更新に用いる遷移の数
		self.qnet = QNetwork(num_state,dropout_rate=dropout_rate)
		self.gamma = 0.99
		self.target_qnet = copy.deepcopy(self.qnet)  # ターゲットネットワーク
		self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr, weight_decay=weight_decay)
		self.replay_buffer = ReplayBuffer(memory_size)
		self.target_update_interval = 100  # ターゲットネットワークのパラメータ更新間隔
		self.steps_done = 0  # エージェントが行った総ステップ数

		# モデルをデバイスに移動
		self.qnet = self.qnet.to(device)
		self.target_qnet = self.target_qnet.to(device)

	def load_model(self, model_path):
		self.qnet.load_state_dict(torch.load(model_path))
		self.target_qnet.load_state_dict(torch.load(model_path))


	# Q関数を更新
	def update_q(self):
		batch = self.replay_buffer.sample(self.batch_size)
		states = np.concatenate(batch["states"], axis=0)  # 状態を適切な形状に整形
		q = self.qnet(torch.tensor(np.float32(states), dtype=torch.float).to(device))

		targetq = copy.deepcopy(q.data.cpu().numpy())

		# Q値が最大の行動だけQ値を更新（最大ではない行動のQ値はqとの2乗誤差が0になる）
		for i in range(self.batch_size):
			target_q_value = batch["rewards"][i]
			targetq[i, batch["actions"][i]] = target_q_value

		self.optimizer.zero_grad()

		# lossとしてMSEを利用
		loss = nn.MSELoss()(q, torch.tensor(targetq).to(device))
		loss.backward()
		self.optimizer.step()

		self.steps_done += 1

		# ターゲットネットワークのパラーメタを更新
		if self.steps_done % self.target_update_interval == 0:
			self.target_qnet = copy.deepcopy(self.qnet)


	# # Q関数を更新
	# def update_q(self):
	# 	batch = self.replay_buffer.sample(self.batch_size)
	# 	states = np.concatenate(batch["states"], axis=0)  # 状態を適切な形状に整形
	# 	# q = self.qnet(torch.tensor(np.float32(states), dtype=torch.float))

	# 	# テンソルをデバイスに移動
	# 	q = self.qnet(torch.tensor(np.float32(states), dtype=torch.float).to(device))

	# 	# targetq = copy.deepcopy(q.data.numpy())
	# 	targetq = copy.deepcopy(q.data.cpu().numpy())


	# 	# DDQNの変更
	# 	next_states = np.concatenate(batch["next_states"], axis=0)  # 状態を適切な形状に整形
	# 	# online_next_q = self.qnet(torch.tensor(np.float32(next_states), dtype=torch.float))

	# 	# target_next_q = self.target_qnet(torch.tensor(np.float32(next_states), dtype=torch.float))

	# 	# テンソルをデバイスに移動
	# 	online_next_q = self.qnet(torch.tensor(np.float32(next_states), dtype=torch.float).to(device))
	# 	target_next_q = self.target_qnet(torch.tensor(np.float32(next_states), dtype=torch.float).to(device))
	# 	online_next_actions = torch.argmax(online_next_q, dim=1)

	# 	# Q値が最大の行動だけQ値を更新（最大ではない行動のQ値はqとの2乗誤差が0になる）
	# 	for i in range(self.batch_size):
	# 		next_state = batch["next_states"][i]

	# 		# Dueling-DDQNを使用してターゲットQ値を計算
	# 		target_q_value = batch["rewards"][i] + self.gamma * target_next_q[i, online_next_actions[i]].item()
	# 		targetq[i, batch["actions"][i]] = target_q_value

	# 	self.optimizer.zero_grad()

	# 	# lossとしてMSEを利用
	# 	# loss = nn.MSELoss()(q, torch.tensor(targetq))
	# 	loss = nn.MSELoss()(q, torch.tensor(targetq).to(device))
	# 	loss.backward()
	# 	self.optimizer.step()

	# 	self.steps_done += 1

	# 	# ターゲットネットワークのパラーメタを更新
	# 	if self.steps_done % self.target_update_interval == 0:
	# 		self.target_qnet = copy.deepcopy(self.qnet)

	def get_greedy_action(self, states, mask):  # statesとmaskは二次元配列

		# テンソルをデバイスに移動
		states_tensor = torch.tensor(states, dtype=torch.float).to(device)
		mask_tensor = torch.tensor(mask, dtype=torch.bool).to(device)

		# Q値の計算
		q_values = self.qnet(states_tensor, mask_tensor).data  # 各馬に対するQ値

		# 最大のQ値を持つ馬を選択
		action = torch.argmax(q_values.mean(dim=0), dim=0).item()  # 各馬のQ値の平均から最大のものを選択
		q_value = q_values[0][action].item()
		
		return action, q_value

	def get_action(self, states, episode, mask):
		epsilon_start = 0.9  # epsilonの初期値を上げる
		epsilon_end = 0.05  # epsilonの最終的な値
		epsilon_decay = 80  # 減衰率を調整するパラメータ
		epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)

		if epsilon <= np.random.uniform(0, 1):
			action, q_value = self.get_greedy_action(states, mask)
			is_random = False  # この行動はgreedyな行動
		else:
			available_actions = np.where(mask == 1)[0]  # マスクされていない行動を取得
			action = np.random.choice(available_actions)  # 有効な行動からランダムに選択
			q_value = None
			is_random = True  # この行動はランダムな行動

		return action, q_value, is_random


	# def get_greedy_action(self, state, mask):  # mask引数を追加

	# 	# テンソルをデバイスに移動
	# 	state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to(device)
	# 	mask_tensor = torch.tensor(mask, dtype=torch.bool).to(device)  # 追加: マスクテンソル
		
	# 	q_values = self.qnet(state_tensor, mask_tensor).data
	# 	action = torch.argmax(self.qnet(state_tensor, mask_tensor).data, dim=1).tolist()  # マスクを入力として渡す
	# 	return action[0],q_values[0][action[0]].item()

	# def get_action(self, state, episode, mask):  # mask引数を追加
	# 	epsilon_start = 0.9  # epsilonの初期値を上げる
	# 	epsilon_end = 0.05  # epsilonの最終的な値
	# 	epsilon_decay = 80  # 減衰率を調整するパラメータ
	# 	epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)

	# 	if epsilon <= np.random.uniform(0, 1):
	# 		action,q_value = self.get_greedy_action(state, mask)  # mask引数を追加
	# 		is_random = False  # この行動はgreedyな行動（自分で選択）
	# 	else:
	# 		available_actions = np.where(mask == 1)[0]  # マスクされていない行動を取得
	# 		action = np.random.choice(available_actions)  # 有効な行動からランダムに選択
	# 		q_value = None
	# 		is_random = True # この行動はランダムな行動
	# 	return action,q_value,is_random


	# def get_action(self, state, episode, mask):  # mask引数を追加
	# 	epsilon = 0.5 * (1 / (episode + 1))

	# 	if epsilon <= np.random.uniform(0, 1):
	# 		action = self.get_greedy_action(state, mask)  # mask引数を追加
	# 	else:
	# 		available_actions = np.where(mask == 1)[0]  # マスクされていない行動を取得
	# 		action = np.random.choice(available_actions)  # 有効な行動からランダムに選択
	# 	return action

from tqdm import tqdm


def print_weights(network):
	for name, param in network.named_parameters():
		print(f"{name}:\n{param}")


def plot_weight_distribution(weight_path):
	weights = torch.load(weight_path)
	# Convert the weights into a single numpy array
	weight_values = np.concatenate([v.cpu().flatten().numpy() for v in weights.values()])
	plt.hist(weight_values, bins=20)
	plt.show()

def print_weight_statistics(weight_path):
	weights = torch.load(weight_path)
	weight_values = np.concatenate([v.flatten().cpu().numpy() for v in weights.values()])

	print(f'Weight statistics for {weight_path}:')
	print(f'Mean: {np.mean(weight_values)}')
	print(f'Median: {np.median(weight_values)}')
	print(f'Standard deviation: {np.std(weight_values)}')
	print(f'Max: {np.max(weight_values)}')
	print(f'Min: {np.min(weight_values)}')
 


 # 以下の部分を修正
num_state = 22

# 各種設定
num_episode = 1000  # 学習エピソード数
memory_size = 1000  # replay_bufferの大きさ
initial_memory_size = 200  # 最初にw貯めるランダムな遷移の数

# ログ
reward_list = []
num_average_episodes = 2

env = Environment(df_all)

agent = DqnAgent(num_state, memory_size=memory_size)
# モデルをデバイスに移動
agent.qnet.to(device)
agent.target_qnet.to(device)
# agent.load_model('dqn_urawa_0425_2')  # 'path/to/'を実際のパスに置き換えてください


# 最初にreplay bufferにランダムな行動をしたときのデータを入れる
for race_id in env.race_id_list[:initial_memory_size]:
	state = env.get_state(race_id)
	action = np.random.randint(state.shape[0])  # ランダムに馬を選択
	reward, bet_money = env.bet_fukushou(race_id, action)
	# next_state = state  # next_stateは現在の状態と同じです（エピソード単位で扱っているため）

	transition = {
		'state': state,
		'reward': reward,
		'action': action,
		'bet_money': bet_money
		# 'next_state': next_state  # next_stateを追加
	}
	agent.replay_buffer.append(transition)


print("random finish")
avg_q_value_list = []  # 平均Q値を保存するためのリストを初期化
std_q_value_list = []  # 平均Q値を保存するためのリストを初期化
num_random_actions = 0
num_greedy_actions = 0

# DQN エージェントの学習部分を調整
# 学習部分の変更
for episode in tqdm(range(num_episode)):
	episode_reward = 0
	q_value_list = []  # Q値を格納するリストを追加
	for race_id in env.race_id_list:
		state = env.get_state(race_id)
		# mask = np.ones(num_horses_in_race)
		num_horses_in_race = len(state)

		mask = np.zeros(12)  # 全ての行動に対して初期マスクを0に設定
		mask[:num_horses_in_race] = 1  # 馬の数だけマスクを1に設定

		# print("--------")
		action,q_value,is_random = agent.get_action(state, episode, mask)  # 全馬の情報から一つの行動を選択
		# print("action:" + str(action))
		if is_random:
			num_random_actions += 1
		else:
			num_greedy_actions += 1
		if q_value is not None:
			q_value_list.append(q_value)

		# action,q_value,is_random = agent.get_action(state, episode, mask)  # 全馬の情報から一つの行動を選択
		# print(action)
		reward, bet_money = env.bet_fukushou(race_id, action)  # 選択した行動による報酬を取得
		# print(reward)
		# print(bet_money)
		# print("********")
		# sys.exit()

		# next_state = env.get_state(race_id)  # 次の状態を取得（全馬の情報）

		# 遷移をリプレイバッファに保存
		transition = {
			'state': state,
			'reward': reward,
			'bet_money': bet_money,
			'action': action
			# 'next_state': next_state
		}
		agent.replay_buffer.append(transition)
		agent.update_q()

		episode_reward += reward  # 報酬を累積

	random_ratio = num_random_actions / (num_random_actions + num_greedy_actions)
	print(f"Episode {episode} Random Ratio: {random_ratio}")

	num_random_actions = 0
	num_greedy_actions = 0

	reward_list.append(episode_reward)
	# 初回のみちゃんと学習できているかどうかチェック
	if episode == 0:
		print_weights(agent.qnet)

	if episode % 10 == 0:
		print("Episode {} finished | Episode reward {}".format(episode, episode_reward))
		# 累積報酬の移動平均を表示
		moving_average = np.convolve(reward_list, np.ones(num_average_episodes) / num_average_episodes, mode="valid")
		# グラフを描画
		plt.plot(moving_average)
		plt.xlabel("Episode")
		plt.ylabel("Moving Average of Episode Rewards")
		plt.show()

		# 保存先のディレクトリを指定
		save_dir = "/content/drive/MyDrive/jra_dqn"  # your_folder_nameを任意のフォルダ名に変更してください。

		# フォルダが存在しない場合は作成
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# 保存ファイル名
		save_path = os.path.join(save_dir, 'dqn_jra_0415pth')
		# モデルの重みを保存
		torch.save(agent.qnet.state_dict(), save_path)
  

		weight_path = os.path.join(weight_dir, f'weights_ep_{episode}.pth')
		torch.save(agent.qnet.state_dict(), weight_path)

		plot_weight_distribution(os.path.join(weight_dir, 'weights_ep_0.pth'))
		plot_weight_distribution(os.path.join(weight_dir, f'weights_ep_{episode}.pth'))
		
		print_weight_statistics(os.path.join(weight_dir, f'weights_ep_{episode}.pth'))


		avg_q_value = np.mean(q_value_list)  # 平均Q値を計算
		std_q_value = np.std(q_value_list)  # ばらつき（標準偏差）を計算

		avg_q_value_list.append(avg_q_value)
		std_q_value_list.append(std_q_value)



		# エピソード終了ごとに報酬とQ値の統計を出力
		# エピソードの報酬をプロット
		plt.figure(figsize=(12, 5))
		plt.subplot(1, 2, 1)
		plt.plot(reward_list)
		plt.title('Episode reward over time')
		plt.xlabel('Episode')
		plt.ylabel('Total reward')

		# 平均Q値をプロット
		plt.subplot(1, 2, 2)
		plt.plot(avg_q_value_list)
		plt.title('Average Q value over time')
		plt.xlabel('Episode')
		plt.ylabel('Average Q value')
		plt.tight_layout()
		plt.show()


		# 平均Q値をプロット
		plt.subplot(1, 2, 2)
		plt.plot(std_q_value_list)
		plt.title('Std Q value over time')
		plt.xlabel('Episode')
		plt.ylabel('Std Q value')
		plt.tight_layout()
		plt.show()





# 累積報酬の移動平均を表示
moving_average = np.convolve(reward_list, np.ones(num_average_episodes) / num_average_episodes, mode="valid")
# グラフを描画
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Moving Average of Episode Rewards")
plt.show()

# モデルの重みを保存
torch.save(agent.qnet.state_dict(), save_path)

print("all done.")
