from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, env=None, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env = env
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        if self.locals['dones']:
            self.logger.record('Dlz/Typ1', self.locals['infos'][0]['Typ1'])
            self.logger.record('Dlz/Typ2', self.locals['infos'][0]['Typ2'])
            self.logger.record('Dlz/Typ3', self.locals['infos'][0]['Typ3'])
            self.logger.record('Dlz/Typ4', self.locals['infos'][0]['Typ4'])
            self.logger.record('Dlz/Typ5', self.locals['infos'][0]['Typ5'])
            self.logger.record('Varianz/Warteschlangen', self.locals['infos'][0]['Warteschlangen'])
            self.logger.record('Varianz/Auslastung', self.locals['infos'][0]['Auslastung'])
            self.logger.record('Mittelwert/Warteschlangen', self.locals['infos'][0]['Av_Warteschlangen'])
            self.logger.record('Mittelwert/Auslastung', self.locals['infos'][0]['Av_Auslastung'])
            self.logger.record('Plan/Anteil_fertigeProdukte', self.locals['infos'][0]['Soll'])
        return True  # return False to stop the training early

    def _on_rollout_start(self) -> None:
        print('Rollout_env_reset')
        self.env.reset()
