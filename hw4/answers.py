r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
from itertools import product

# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp['batch_size'] = 16
    hp['gamma'] = 0.99
    hp['beta'] = 0.5
    hp['learn_rate'] = 1e-3
    hp['eps'] = 1e-8
    hp['num_workers'] = 1
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    
    # ====== YOUR CODE: ======
    hp['batch_size'] = 8
    hp['gamma'] = 0.99
    hp['beta'] = 0.5
    hp['delta'] = 1.0
    hp['learn_rate'] = 1e-3
    hp['eps'] = 1e-8
    hp['num_workers'] = 2
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**
By subtracting a baseline, mean value of the policy weights, and therefore, changing the rewards from being an absolute value, to some relative value, we are measuring how good 
an action is, but are not adding importance to the absolute value. 
If all the rewards are positve, we will increase all the values together, but the currently better action will get a much bigger boost. 
by changing to the baseline value, we measure how relativly better an action is than the opposing actions, and therefore, increase the 'gap'
between currently good and currently bad actions. this will result in a much lower variance than without baseline value.
For example, for a game where all scores will be positive, the only ability to change the resulting probabilities of the actions, is increasing the scores of each action.
resulting in the effect where without baseline, we will get a distribution resulting from a softmax, where the enthropy will be very low (a delta essentially), where
with baseline, the distribution will be more evenly distributed, but where the better action will be relativly rewarded, as we wanted.
"""


part1_q2 = r"""
**Your answer:**
$v_\pi(s)$ is given to us with $q_\pi(s,a)$ such that $v_\pi(s) = q_\pi(s,a) - advantage$. the purpose of the advantage is to give us indication,
how much more 'advantageous' it is to partake in a specific action, instead of all the opposing ones. this is estimated with the critic generating a score as to see what action it decides is the most advantageous, and forcing the actor along that path. 
the reason we are estimating using q-value regression is that the q-values approximate the environment on which the critic 'learns' what actions are more advantageous than others,
and therefore, allows us to see which action is not only good, but will typically lead to better actions after that, by learning on the history of those q-values.
as seen in q1, this also stabilises the model, which is great in order to train it better. 
"""


part1_q3 = r"""
**Your answer:**
1. As described in q1 and q2, we will iterate on each graph al illustrate according to the answers we gave.
**Loss_p** 
we can see that using baseline loss results in a graph that is almost constantly zero, therefore, it decreases the policy variance to be very little, for better or for worse. this results in a loss function that takes the loss of the policy to a very minute effect, while without baseline, it has a much larger effect on the loss functions in total.
```
**Baseline**
we can see that the enthropy loss doesn't effect the baseline loss graph, as expected, and both perform almost exactly the same

**Loss_e**
here, we can see that combining baseline and enthropy is beneficial to the enthropy loss, and results in a lower entropy loss, as described in q1.

**mean_rewards** 
totalling everything together, the baseline graphs are better, and the entropy and non entropy results are very close to each other, with some std that goes either way. it usually goes the way that entropy loss is a better, more fitting loss.

2. *AAC*
The actor critic graph trains very quickly to be at a very good point of performance, but has some variance and noise, which might be as a result of training 2 networks 'with' each other.
in the **Loss_P** graph, we can see that we reach a positive policy loss very quickly, and are relatively stable at that point, which we think might say that the policy route we have taken is a good route to take.
the entropy loss is the best between all losses resulted as of yet, and the mean rewards graph is very competative with great results we have already had with our classic networks. we think that more ephocs, a slightly better net, or better hyper params would result in a significantly better net.
```
An equation: $e^{i\pi} -1 = 0$

"""
