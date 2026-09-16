

def test_shell_hardening_variables_never_reach_a_job():
    """An agent shell's NoDefaultCurrentDirectoryInExePath broke job 1461.

    Inherited by a job, it makes CreateProcess refuse `.venv/Scripts/python.exe`.
    """
    from workerq.util import scrub_inherited_env

    env = {"NoDefaultCurrentDirectoryInExePath": "1", "NODEFAULTCURRENTDIRECTORYINEXEPATH": "1", "PATH": "x"}
    assert scrub_inherited_env(env) == {"PATH": "x"}
