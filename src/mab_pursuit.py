import numpy as np
import matplotlib.pyplot as plt
import sys
import math

# set the parameters
TIMES = 500
EPISODES = 5000
K = 15

# bandit
def mab_pursuit(P_MIN, BETA, ALPHA):
    rewards_all_sa = np.zeros((TIMES, EPISODES))
    best_action_freq_sa = np.zeros((TIMES, EPISODES))

    for r in range(TIMES):
        q_star_sa = np.zeros(K)
        q_a_sa = np.zeros(K)
        action_freq_sa = np.zeros(K)

        prob_list = np.zeros(K)
        arm_list = list(range(K))
        arm_list = np.array(arm_list)

        

        for t in range(EPISODES):

            # add the noise
            for i in range(K):
                noise = np.random.normal(0.0, 0.01)
                q_star_sa[i] += noise
            
            # find the best action in this (status, action) as reference
            best_action_sa = np.argmax(q_star_sa)
            if t == 0:
                pre_best_action_sa = best_action_sa
            #####################################################

            # update the probability list
            flag = True

            if t != 0:
                P_MAX = 1 - (K - 1) * P_MIN
                for i in range(K):
                    if i != pre_best_action_sa:
                        prob_list[i] = prob_list[i] + ALPHA * (P_MIN - prob_list[i])
                    else:
                        prob_list[i] = prob_list[i] + ALPHA * (P_MAX - prob_list[i])
            else:
                prob_list[pre_best_action_sa] = 1.0

            cur_prob_sum = np.sum(prob_list)
            if (abs(cur_prob_sum - 1.0)) > 1e-6:
                flag = False

            # choose the action in this step
            action_sa = action_sa = np.random.randint(0, K)
            if flag:
                action_sa = np.random.choice(a = arm_list, p = prob_list)
            
            #####################################################

            # get the reward and update
            reward_sa = np.random.normal(q_star_sa[action_sa], 1)
            q_a_sa[action_sa] += (reward_sa - q_a_sa[action_sa]) * BETA

            # update the optimallity
            rewards_all_sa[r, t] = reward_sa

            if action_sa == best_action_sa:
                best_action_freq_sa[r, t] = 1
            
            # greedy action of the previous step
            list_max_sa = []
            q_max_sa = np.max(q_a_sa)
            for i in range(K):
                if q_a_sa[i] == q_max_sa:
                    list_max_sa.append(i)
            pre_best_action_sa = np.random.choice(list_max_sa)
    
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
ax.set_title("adaptive pursuit")

test_list = [[0.001, 0.3, 0.3], [0.001, 0.1, 0.3], [0.001, 0.3, 0.1], [0.01, 0.3, 0.1]]

for one_test in test_list:
    cur_label = "P_min = " + str(one_test[0]) + ", beta = " + str(one_test[1]) + ", alpha = " + str(one_test[2])
    cur_res = mab_pursuit(one_test[0], one_test[1], one_test[2])
    ax.plot(plt_x_axis, cur_res, label = cur_label)

ax.legend()
plt.show()
