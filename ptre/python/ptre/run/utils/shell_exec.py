import multiprocessing
import subprocess
import sys
import threading

def _exec_command(command, stdout, stderr):
  executor_shell = subprocess.Popen(command, shell=True, stdout=stdout,
      stderr=stderr)
  exit_code = executor_shell.wait()
  sys.exit(exit_code)

def execute(command, env=None, stdout=None, stderr=None):
  print(command)
  parent_stdout, child_stdout = multiprocessing.Pipe()
  parent_stderr, child_stderr = multiprocessing.Pipe()
  p = multiprocessing.Process(target=_exec_command,
                              args=(command, child_stdout, child_stderr))
  p.start()
  #print("rank_0: " + parent_stderr.recv())
  #print("rank_0: " + parent_stdout.recv())
  p.join()

'''
  ctx = multiprocessing.get_context('spawn')
  exit_event = ctx.Event()

  (stdout_r, stdout_w) = ctx.Pipe()
  (stderr_r, stderr_w) = ctx.Pipe()
  (r, w) = ctx.Pipe()

  middleman = ctx.Process(target=_exec_middleman, args=(command, env,
      exit_event, (stdout_r, stdout_w), (stderr_r, stderr_w), (r, w)))

  middleman.start()

  print("middleman started")
  r.close()
  stdout_w.close()
  stderr_w.close()

  if stdout is None:
    stdout = sys.stdout
  if stderr is None:
    stderr = sys.stderr

  stdout_fwd = in_thread
'''
