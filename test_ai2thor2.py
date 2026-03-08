import os, sys, time

os.environ['DISPLAY'] = ':0'
LOGFILE = '/home/march/workspace/Yaning/VAGEN-eb-alfred/thor_test_result2.txt'

def log(msg):
    with open(LOGFILE, 'a') as f:
        f.write(f'[{time.time():.1f}] {msg}\n')
    print(msg, flush=True)

with open(LOGFILE, 'w') as f:
    f.write('')

log('START')
try:
    import embodiedbench.envs.eb_alfred.EBAlfEnv as ebalfenv_mod
    ebalfenv_mod.X_DISPLAY = '0'
    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv

    t0 = time.time()
    log('Creating EBAlfEnv...')
    env = EBAlfEnv(eval_set='base', resolution=500)
    log(f'Created in {time.time()-t0:.1f}s. {env.number_of_episodes} episodes.')

    env._current_episode_num = 0
    task = env.dataset[0]

    from embodiedbench.envs.eb_alfred import utils
    traj_data = utils.load_task_json(task)
    log(f'Task: {task["instruction"][:60]}')

    scene_num = traj_data['scene']['scene_num']
    scene_name = 'FloorPlan%d' % scene_num
    log(f'Calling env.env.reset({scene_name})...')

    import threading
    result = [None]
    error = [None]

    def do_reset():
        try:
            result[0] = env.env.reset(scene_name)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=do_reset)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        log(f'TIMEOUT: env.env.reset({scene_name}) hung for 30s!')
        os._exit(1)
    elif error[0]:
        log(f'RESET ERROR: {error[0]}')
    else:
        log(f'Reset done in {time.time()-t0:.1f}s!')

    env.close()
    log('SUCCESS')
except Exception as e:
    import traceback
    log(f'ERROR: {e}')
    log(traceback.format_exc())
