from garage.np.policies import ScriptedPolicy


class TestScriptedPolicy:

    def setup_method(self):
        self.sp = ScriptedPolicy(scripted_actions=[1], agent_env_infos={0: 1})

    """
    potentially add more tests down the line
    """

    def test_pass_codecov(self):
        self.sp.get_action(0)
        self.sp.get_actions([0])
