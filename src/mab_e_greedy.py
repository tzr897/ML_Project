import numpy as np
import matplotlib.pyplot as plt
import sys

# set the parameters
TIMES = 500
EPISODES = 5000
K = 15

# bandit
def mab_e_greedy(epsilon, BETA):
    rewards_all_sa = np.zeros((TIMES, EPISODES))
    best_action_freq_sa = np.zeros((TIMES, EPISODES))

    for r in range(TIMES):
        q_star_sa = np.zeros(K)
        q_a_sa = np.zeros(K)
        action_freq_sa = np.zeros(K)

        for t in range(EPISODES):

            # add the noise
            for i in range(K):
                noise = np.random.normal(0, 0.01)
                q_star_sa[i] += noise
            
            # find the best action in this (status, action) as reference
            best_action_sa = np.argmax(q_star_sa)
            
            #####################################################
            # choose the action in this step
            action_sa = 0
            prob = np.random.rand()

            if prob < epsilon:
                action_sa = np.random.randint(0, K)
            else:
                list_max_sa = []
                q_max_sa = np.max(q_a_sa)
                for i in range(K):
                    if q_a_sa[i] == q_max_sa:
                        list_max_sa.append(i)
                action_sa = np.random.choice(list_max_sa)
            #####################################################

            # get the reward and update
            reward_sa = np.random.normal(q_star_sa[action_sa], 1)
            q_a_sa[action_sa] += (reward_sa - q_a_sa[action_sa]) * BETA

            # update the optimallity
            rewards_all_sa[r, t] = reward_sa

            if action_sa == best_action_sa:
                best_action_freq_sa[r, t] = 1
    
    # calculating
    ratio_o_t_sa = list(range(EPISODES))

    for t in range(EPISODES):
        sum_o_sa = 0
        for r in range(TIMES):
            sum_o_sa += best_action_freq_sa[r, t]
        ratio_o_t_sa[t] = float(sum_o_sa) / float(TIMES)
    
    return ratio_o_t_sa
    
fig, ax = plt.subplots()
plt.ylim((0, 1))
plt_x_axis = list(range(EPISODES))
ax.set_xlabel("number of episodes")
ax.set_ylabel("optimality")
ax.set_title("e-greedy")


test_list = [[0.01, 0.1], [0.01, 0.3], [0.05, 0.3], [0.1, 0.1]]

for one_test in test_list:
    cur_label = "e = " + str(one_test[0]) + ", beta = " + str(one_test[1])
    cur_res = mab_e_greedy(one_test[0], one_test[1])
    ax.plot(plt_x_axis, cur_res, label = cur_label)

ax.legend()
plt.show()
    