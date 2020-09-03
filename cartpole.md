## CartPole

[동영상자료](https://codetorial.net/articles/cartpole/_videos/original.mp4)

막대기가 제대로 서있으면 매 시간 스텝마다 +1 보상

# 에피소드 종료 조건

- 막대기가 수직으로부터 12도 이상 기울어짐 (-12도 ~ 12도)
- 카트가 중심으로부터 2.4 이상 벗어남 (-2.4 ~ 2.4)
- 시간 스텝이 200보다 커짐 (CartPole-v1의 경우 500)
- 또한 100 번의 연속적인 시도에서 평균 195.0 이상의 점수 (보상)을 얻으면, CartPole-v0 게임을 해결했다고 정의함

# gym 설치

1. 새로운 가상환경 'gym' 만들어주기

```
conda create -n gym python=3.5 anaconda
```

2. 가상환경 activate

```
activate gym
```

3. gym 모듈 설치

```
pip install gym
```

# 환경

CartPole-v0 환경의 인스턴스를 1000회의 시간 스텝 동안 실행하고, 각 스텝에서의 환경을 렌더링

```python
import gym

env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # 랜덤 액션을 취하도록
env.close()
```

# env.reset()

[카트의 위치, 카트의 속도, 막대기의 각도, 막대기의 회전율] 반환

```python
observation = env.reset()

print(observation)
```

```
[-0.03708528 -0.02401037  0.00454671 -0.03939116]
```

# env.action_space

게임 환경에서 선택할 수 있는 행동(action_space)중 하나의 값 샘플링. 언제나 0 또는 1 값 출력

```
action = env.action_space.sample()

print(action)
```

# env.step

action을 선택했을 때, (observation, reward, done, info)반환

```python
env = gym.make('CartPole-v0')
observation = env.reset()
action = env.action_space.sample()
step = env.step(action)

print('First observation:', observation)
print('Action:', action)
print('Step:', step)
```
