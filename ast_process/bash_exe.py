from subprocess import Popen
from threading import Timer
import os


def execute_command(command_list, cwd, stdout_path='stdout.txt', stderr_path='stderr.txt'):
    """
    Execute the command in the command list using the shell interface.
    Basically, execute shell command using python
    Default: Saving output to tmp/stderr.txt and tmp/stdout.txt
    """
    command = command_list
    current_dir = os.getcwd()
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    stdout_txt = os.path.join(current_dir, 'tmp', stdout_path)
    stderr_txt = os.path.join(current_dir, 'tmp', stderr_path)
    kill = lambda process: process.kill()
    with open(stdout_txt, "wb") as out, open(stderr_txt, "wb") as err:
        p1 = Popen(command, stdout=out, stderr=err, cwd=cwd, shell=False)
        my_timer = Timer(100, kill, [p1])
        try:
            my_timer.start()
            p1.communicate()
        finally:
            my_timer.cancel()
        print(p1.returncode)
