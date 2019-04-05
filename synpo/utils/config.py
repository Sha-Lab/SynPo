class Config:
  def __init__(self):
    self.task_fn = None
    self.optimizer_fn = None
    self.network_fn = None
    self.policy_fn = None
    self.replay_fn = None
    self.discount = 0.99
    self.target_network_update_freq = 100
    self.exploration_steps = 50000
    self.logger = None
    self.history_length = 4
    self.test_interval = 100
    self.test_repetitions = 10
    self.double_q = False
    self.tag = 'vanilla'
    self.update_interval = 1
    self.action_shift_fn = lambda a: a
    self.reward_shift_fn = lambda r: r
    self.episode_limit = 0
    self.save_interval = 0
    self.max_steps = 0
    self.max_eps   = 200000
    self.grad_clip = 0
    self.n_test_samples = 100
    self.value_loss_weight = 0.5
    self.one_traj = False
    self.extend = None
