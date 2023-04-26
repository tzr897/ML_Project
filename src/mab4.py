import numpy as np
import matplotlib.pyplot as plt
import sys
import math

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

def mab_softmax(TEMPERATURE, BETA):
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
                noise = np.random.normal(0, 0.01)
                q_star_sa[i] += noise
            
            # find the best action in this (status, action) as reference
            best_action_sa = np.argmax(q_star_sa)
            
            #####################################################

            # update the probability list
            denominator = 0.0
            for i in range(K):
                denominator += math.exp(q_a_sa[i] / TEMPERATURE)
            
            for i in range(K):
                numerator = math.exp(q_a_sa[i] / TEMPERATURE)
                prob_list[i] = numerator / denominator

            # choose the action in this step
            action_sa = np.random.choice(a = arm_list, p = prob_list)
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

def mab_prob_matching(P_MIN, BETA):
    rewards_all_sa = np.ones((TIMES, EPISODES))
    best_action_freq_sa = np.zeros((TIMES, EPISODES))

    for r in range(TIMES):
        q_star_sa = np.ones(K)
        q_a_sa = np.ones(K)
        action_freq_sa = np.zeros(K)

        prob_list = np.zeros(K)
        arm_list = list(range(K))
        arm_list = np.array(arm_list)

        for t in range(EPISODES):

            # add the noise
            for i in range(K):
                noise = np.random.normal(1.0, 0.01)
                q_star_sa[i] += noise
            
            # find the best action in this (status, action) as reference
            best_action_sa = np.argmax(q_star_sa)
            
            #####################################################

            # update the probability list
            flag = True
            denominator = np.sum(q_a_sa)
            if denominator <= 0.0:
                flag = False
            for i in range(K):
                if (q_a_sa[i] / denominator) <= 0.0:
                    flag = False
                    break
                prob_list[i] = P_MIN + (1 - K * P_MIN) * (q_a_sa[i] / denominator)

            # choose the action in this step
            action_sa = 0
            if flag:
                action_sa = np.random.choice(a = arm_list, p = prob_list)
            else:
                action_sa = np.random.choice(a = arm_list, p = prob_list)
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
            if t != 0:
                P_MAX = 1 - (K - 1) * P_MIN
                for i in range(K):
                    if i != pre_best_action_sa:
                        prob_list[i] = prob_list[i] + ALPHA * (P_MIN - prob_list[i])
                    else:
                        prob_list[i] = prob_list[i] + ALPHA * (P_MAX - prob_list[i])
            else:
                prob_list[pre_best_action_sa] = 1.0


            # choose the action in this step
            action_sa = np.random.choice(a = arm_list, p = prob_list)
            
            #####################################################

            # get the reward and update
            reward_sa = np.random.normal(q_star_sa[action_sa], 1)
            q_a_sa[action_sa] += (reward_sa - q_a_sa[action_sa]) * BETA

            # update the optimallity
            rewards_all_sa[r, t] = reward_sa

            if action_sa == best_action_sa:
                best_action_freq_sa[r, t] = 1
            
            pre_best_action_sa = best_action_sa
    
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
ax.set_title("adhoc methods")

label1 = "e-greedy, e = 0.01, beta = 0.1"
res1 = mab_e_greedy(0.01, 0.1)
ax.plot(plt_x_axis, res1, label = label1)

label2 = "softmax, T = 0.3, beta = 0.3"
res2 = mab_softmax(0.3, 0.3)
ax.plot(plt_x_axis, res2, label = label2)

label3 = "pursuit, P_min = 0.001, alpha = 0.3, beta = 0.3"
res3 = mab_pursuit(0.001, 0.3, 0.3)
ax.plot(plt_x_axis, res3, label = label3)

label4 = "probability matching, P_min = 0.0001, beta = 0.5"
res4 = mab_prob_matching(0.0001, 0.5)
ax.plot(plt_x_axis, res4, label = label4)

ax.legend()
plt.show()
    