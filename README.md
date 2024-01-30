# OpenAI-GYM-RL-Robo-Taxi-Driving

Robo-Taxi Driving reinforcement learning project using Q-Learning and SARSA algorithms with OpenAI Gym. This project provides a hands-on experience in training a virtual autonomous taxi to navigate through a dynamic urban environment.



## Libraries Required:
Make sure you have the following libraries installed before running the code:

OpenAI Gymnasium 
NumPy
Pygame
Matplotlib
Jupyter Notebook (for running the notebook file)
You can install the required libraries using the following command:
```
pip install gymnasium pygame numpy matplotlib jupyter

```
## How to run Code 
1 Clone the Repository:
```
gh repo clone vaibhav720/OpenAI-GYM-RL-Robo-Taxi-Driving

```
2 Launch Jupyter Notebook:
```
jupyter notebook

```

3 Open the notebook file:

* Navigate to the notebooks directory.
* Launch the RoboTaxi_RL.ipynb file.
* 
4 Run the Jupyter Notebook cells:

* Execute each cell step by step to train the Robo-Taxi using Q-Learning and SARSA algorithms.
* Follow the comments and instructions provided in the notebook for a clear understanding of the code.
5 Alternatively, run the Python script:
* Execute the RoboTaxi_RL.py script in a Python environment:
```
python VaibhavParikh_Ex1.py

```
## Q Learning Algorithm:-

It is a reinforcement learning algorithm and does not require any environment of model. It is an iterative process updates model in the while exploration[1].
	Q(s,a)=(1- ∝)*Q(s,a)+ ∝ *(r+γ*〖max〗_(a^(' ) ) Q(s^',a^'))
Where, 
Q(s,a) here Q table stores the reward for the particular  position
∝ it is learning rate
r is the rewards
γ it is discount factor
s it is the present position
a’ it is the action
### Output:- 
![image](https://github.com/vaibhav720/OpenAI-GYM-RL-Robo-Taxi-Driving/assets/56918464/49393031-c156-4a52-afff-7fa54ba1d2ab)

![image](https://github.com/vaibhav720/OpenAI-GYM-RL-Robo-Taxi-Driving/assets/56918464/277e5620-0da1-409a-82e0-47c29ec1c66e)

![image](https://github.com/vaibhav720/OpenAI-GYM-RL-Robo-Taxi-Driving/assets/56918464/a942b393-5d5d-4369-9e30-3a45a8208c8c)

## SARSA Algorithm:-
It is also known as the State Action Reward State Action is based on the Markov Decision process policy[2]. It is also known as on policy learning algorithm.

### Output:-
![image](https://github.com/vaibhav720/OpenAI-GYM-RL-Robo-Taxi-Driving/assets/56918464/f87a326a-47bf-4787-9c84-740d930d5c40)
![image](https://github.com/vaibhav720/OpenAI-GYM-RL-Robo-Taxi-Driving/assets/56918464/9b6751e3-6437-4c5e-a851-260319f64c7e)

## Analysis of both Algorithms

![image](https://github.com/vaibhav720/OpenAI-GYM-RL-Robo-Taxi-Driving/assets/56918464/eee9ef08-3380-4b9c-92f3-5d981b012b7e)

From the Comparison of the we can see that Q learning algorithm is much better than the SARSA algorithm and because it gives much more rewards and performs better consistently. While sometimes the SARSA algorithm fails and does not work even after undergoing the dame training rate and proper tuning.


Time taken to train the Q Learning Algorithm  6.920063734054565
Time taken to train The SARSA Algorithm  3.286423683166504

But the Q Learning Algorithm takes much more time for the training and sometimes it take the twice the time and the SARSA algorithm. 




