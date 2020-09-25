from multiprocessing import Pool
import subprocess
import time


def call_job(cmd):
  process = subprocess.Popen(cmd, shell=True)
  print('Excecuting: %s', cmd)
  process.wait()
  out, err = process.communicate()
  if out:
    print('Success: %s' % cmd, out)
  if err:
    print('Fail: %s' % cmd, err)
  return (out, err)


if __name__ == "__main__":
  total_tasks = 300
  # only 2 GPUs, cudaid=0/1
  py_path = 'python /home/chli4934/ubCodeLab/oc_hrl_pytorch/mp_jobs.py --i=%d --cudaid=%d'
  cudaid = -1
  cmd_list = []
  for i in range(total_tasks):
    # cudaid = 0 if cudaid else 1
    cmd_list.append(py_path % (i, cudaid))

  with Pool(processes=18) as pool:
    start = time.time()
    for x in pool.imap(call_job, cmd_list):
      print("(Time elapsed: {}s)".format(int(time.time() - start)))
