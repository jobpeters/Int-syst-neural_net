from kaggle_environments import evaluate, make
import IPython


# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


def run():
    env = make("connectx", debug=True)
    env.render()
    env.reset()
    # Play as the first agent against default "random" agent.
    env.run([my_agent, "random"])
    env.render(width=500, height=450)

    # Play as first position against random agent.
    trainer = env.train([None, "random"])

    observation = trainer.reset()

    while not env.done:
        my_action = my_agent(observation, env.configuration)
        print("My Action", my_action)
        env.render(width=500, height=450)
        observation, reward, done, info = trainer.step(my_action)
        # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
    env.render()

if __name__ == "__main__":
    run()