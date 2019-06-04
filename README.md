# Reinforcement Toys

**This repository proposes an introduction du Q-Learning on some toy exemples**

## Instalation

The notebooks are readable on _GitHUb_ but some features cannot be displayed on the site and it is better to experiment with the algorithms and the parameters.
You can directly _clone_ the repository and install the packages listed in the `requirements.txt` file. Here are the commands using `Anaconda`:
```
git clone https://github.com/JbRemy/Reinforcement_Toys
conda create -n <NAME> python=3.7
conda activate <NAME>
pip install -r requirements.txt
```
For Mac OsX you might also need to install `ffmpeg`: 
```
brew install ffmpeg
```
Note that this is usefull to replicate the exact setting I use to develop and test the repository. However there is not much specific interaction and it might work perfectly with a lot of configuration as long as you have the libraries listed in the `requirements.txt` file.

##  Description

Each notebook presents some teoritical concepts and a new application. Here we describe the notebooks in the order they are meant to be read.

### `IntroGridWorld`

This notebook proposes a theoritical introducation to _Q-Learning_, a short presentation of the classes I implemented to make the experimentationo seamless to the reader and application of the basic _Q-Learning_ algorithm on *maze* problem.

### `CartPole`

This notebook introduces the discretization of the _Q-table_ and demonstrates the importance of the hyper-parameters. The problem is applied to the `CartPole-v1` environment of the [Gym AI](https://gym.openai.com) library.


## TODO

* Verify and update the description of the classes in the first notebook
* Add explration vs exploitation trade-off visualization on the first task
* Add more complex tasks, and algorithms (DeepQ, Lunar lander)
* Complete README for codes
* Update Notebook description (images)

