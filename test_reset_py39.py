import os, time, sys
os.environ['DISPLAY'] = ':0'

LOG = '/home/march/workspace/Yaning/VAGEN-eb-alfred/reset_test_py39.log'

def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'[{time.time():.1f}] {msg}\n')
        f.flush()

with open(LOG, 'w') as f:
    f.write('')

log('START')

try:
    import embodiedbench.envs.eb_alfred.EBAlfEnv as m
    m.X_DISPLAY = '0'
    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv

    t0 = time.time()
    log('Creating EBAlfEnv...')
    env = EBAlfEnv(eval_set='base', resolution=500)
    log(f'Created in {time.time()-t0:.1f}s ({env.number_of_episodes} episodes)')

    env._current_episode_num = 0
    log('Calling env.reset()...')
    obs = env.reset()
    log(f'RESET DONE in {time.time()-t0:.1f}s!')
    log(f'Task: {env.episode_language_instruction[:80]}')

    env.close()
    log(f'SUCCESS total={time.time()-t0:.1f}s')
except Exception as e:
    import traceback
    log(f'ERROR: {e}')
    log(traceback.format_exc())
