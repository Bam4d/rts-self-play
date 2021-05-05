import gym
from collections import Counter
from griddly import GymWrapperFactory, gd
from griddly.RenderTools import RenderToFile
from griddly.util.wrappers import ValidActionSpaceWrapper


class EventFrequencyTracker():

    def __init__(self, window_size):
        self._steps = 0

        self._window_size = window_size

        self._frequency_trackers = [Counter() for _ in range(window_size)]

    def process(self, events):
        for e in events:
            action_name = e['ActionName']
            self._frequency_trackers[-1][action_name] += 1

            if action_name == 'build_barracks':
                print('barracks placed')

            if action_name == 'build_combat':
                print('combat build')

        self._frequency_trackers.pop(0)
        self._frequency_trackers.append(Counter())

    def get_frequencies(self):
        event_totals = Counter()
        for tracker in self._frequency_trackers:
            for key, value in tracker.items():
                event_totals[key] += value

        event_averages = {}
        for k, v in event_totals.items():
            event_averages[k] = v / self._window_size

        return event_totals


if __name__ == '__main__':
    wrapper = GymWrapperFactory()

    wrapper.build_gym_from_yaml("GriddlyRTS-Adv",
                                'griddly_rts.yaml',
                                global_observer_type=gd.ObserverType.ISOMETRIC,
                                player_observer_type=gd.ObserverType.VECTOR,
                                level=0)

    env_original = gym.make(f'GDY-GriddlyRTS-Adv-v0')
    # env_original = gym.make(f'GDY-GriddlyRTS-Adv-v0')

    env_original.reset()
    env_original.enable_history()

    env = ValidActionSpaceWrapper(env_original)

    event_tracker = EventFrequencyTracker(10)

    renderer = RenderToFile()

    for i in range(100000):
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        event_tracker.process(info['History'])

        global_obs = env.render(observer='global', mode='rgb_array')

        #renderer.render(global_obs, 'rts_.png')

        #print(event_tracker.get_frequencies())

        if done:
            env.reset()

    # env.reset()
