import os
import math
from datetime import datetime
import pytz

ALL_SEEDS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
             11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

RUNS_DIR = "/home-nfs/hung/gen/runs"
LOGS_DIR = "/share/data/ripl/hung/a3rl_workspace/logs/a3rl"
HOME_DIR = "/home-nfs/hung"
CHICAGO_TZ = pytz.timezone('America/Chicago')

ENVS = {
    "mujoco/halfcheetah/expert-v0": ("halfcheetah", "e"),
    "mujoco/halfcheetah/medium-v0": ("halfcheetah", "m"),
    "mujoco/halfcheetah/simple-v0": ("halfcheetah", "s"),
    "mujoco/walker2d/expert-v0": ("walker2d", "e"),
    "mujoco/walker2d/medium-v0": ("walker2d", "m"),
    "mujoco/walker2d/simple-v0": ("walker2d", "s"),
    "mujoco/hopper/expert-v0": ("hopper", "e"),
    "mujoco/hopper/medium-v0": ("hopper", "m"),
    "mujoco/hopper/simple-v0": ("hopper", "s"),
    "mujoco/inverteddoublependulum/expert-v0": ("inverteddoublependulum", "e"),
    "mujoco/inverteddoublependulum/medium-v0": ("inverteddoublependulum", "m"),
    "mujoco/ant/expert-v0": ("ant", "e"),
    "mujoco/ant/medium-v0": ("ant", "m"),
    "mujoco/ant/simple-v0": ("ant", "s"),
    "mujoco/swimmer/expert-v0": ("swimmer", "e"),
    "mujoco/swimmer/medium-v0": ("swimmer", "m"),
    "mujoco/reacher/expert-v0": ("reacher", "e"),
    "mujoco/reacher/medium-v0": ("reacher", "m"),
    "D4RL/pen/expert-v2": ("pen", "e"),
    "D4RL/pen/cloned-v2": ("pen", "c"),
    "D4RL/relocate/expert-v2": ("relocate", "e"),
    "D4RL/relocate/cloned-v2": ("relocate", "c"),
    "D4RL/door/expert-v2": ("door", "e"),
    "D4RL/door/cloned-v2": ("door", "c")
}

ENVS_HOPPER = ["mujoco/hopper/expert-v0", "mujoco/hopper/medium-v0",  "mujoco/hopper/simple-v0"]
ENVS_INVERTEDDOUBLEPENDULUM = ["mujoco/inverteddoublependulum/expert-v0", "mujoco/inverteddoublependulum/medium-v0"]
ENVS_HALFCHEETAH = ["mujoco/halfcheetah/expert-v0", "mujoco/halfcheetah/medium-v0",  "mujoco/halfcheetah/simple-v0"]
ENVS_SWIMMER = ["mujoco/swimmer/expert-v0", "mujoco/swimmer/medium-v0"]
ENVS_WALKER2D = ["mujoco/walker2d/expert-v0", "mujoco/walker2d/medium-v0",  "mujoco/walker2d/simple-v0"]
ENVS_ANT = ["mujoco/ant/expert-v0", "mujoco/ant/medium-v0",  "mujoco/ant/simple-v0"]
ENVS_REACHER = ["mujoco/reacher/expert-v0", "mujoco/reacher/medium-v0"]

ENVS_PEN = ["D4RL/pen/expert-v2", "D4RL/pen/cloned-v2"]
ENVS_RELOCATE = ["D4RL/relocate/expert-v2", "D4RL/relocate/cloned-v2"]
ENVS_DOOR = ["D4RL/door/expert-v2", "D4RL/door/cloned-v2"]

ENVS_LIST = {
    "hopper": ENVS_HOPPER,
    "inverteddoublependulum": ENVS_INVERTEDDOUBLEPENDULUM,
    "halfcheetah": ENVS_HALFCHEETAH,
    "swimmer": ENVS_SWIMMER,
    "walker2d": ENVS_WALKER2D,
    "ant": ENVS_ANT,
    "reacher": ENVS_REACHER,
    "pen": ENVS_PEN,
    "relocate": ENVS_RELOCATE,
    "door": ENVS_DOOR
}

ENV_CORE = list(ENVS_LIST.keys())
def get_env_name(env_core, difficulty):
    a = [key for key, value in ENVS.items() if value == (env_core, difficulty)]
    if len(a) == 0:
        print('No environment as such')
    elif len(a) > 1:
        print('Multiple environment matches. Something is wrong with ENVS')
    else:
        return a[0]

def get_time():
    c = datetime.now(CHICAGO_TZ)
    return c.strftime("%y%m%d_%H%M%S"), c.strftime("%m/%d/%y %H:%M:%S")


#! BATCH PARAMETERS
# TODO: choose variant: "rlpd", "a3rl", "a3rl_5to5", "a3rl_20to5", "a3rl_5to5_noanneal", "a3rl_noanneal", "sacfd"
variant = "a3rl_noanneal"
list_env_name = [
                    # ("halfcheetah", 1, 0, 0), # done for first round of sacfd
                #  ("halfcheetah", 1, 0.05, 0),
                #  ("halfcheetah", 1, 0.1, 0),
                #   ("ant", 1, 0, 0),
                #  ("ant", 1, 0.05, 0),
                #  ("ant", 1, 0.1, 0),
                #  ("walker2d", 1, 0, 0),
                #  ("walker2d", 1, 0.05, 0),
                #  ("walker2d", 1, 0.1, 0),
                #  ("hopper", 1, 0, 0),
                #  ("hopper", 1, 0.05, 0),
                #  ("hopper", 1, 0.1, 0),
                 
                 ("pen", 0, 1, 0),
                 ("pen", 0, 1, 0.05),
                 ("pen", 0, 1, 0.1),
                 ("relocate", 0, 1, 0),
                 ("relocate", 0, 1, 0.05),
                 ("relocate", 0, 1, 0.1),
                 ("door", 0, 1, 0),
                 ("door", 0, 1, 0.05),
                 ("door", 0, 1, 0.1)
                ]
list_a = [0.3]
list_l = [0.3]
list_a_l = [(0.3, 0.3), (0.6, 0.03)]
list_seed = ALL_SEEDS[:3]
grid_search = False
version = "v10"
#! END OF BATCH PARAMETERS

if grid_search:
    list_a_l = [(a, l) for a in list_a for l in list_l]
if variant in ["rlpd", "sacfd"]:
    list_a_l = [(0, 0)]

VARIANTS = {
    "sacfd": {
        "offline_ratio": 1.0,
        "use_density": False,
        "heuristics": "none",
        "start_a3rl": 300000,
        "use_alternate": False
    },
    "a3rl": {
        "offline_ratio": 0.5,
        "use_density": True,
        "heuristics": "adv",
        "start_a3rl": 100000,
        "use_alternate": False,
        "h_alpha_final_p": 1,
        "h_beta_i": 0.4,
        "h_beta_f": 1.0
    },
    "a3rl_noanneal": {
        "offline_ratio": 0.5,
        "use_density": True,
        "heuristics": "adv",
        "start_a3rl": 100000,
        "use_alternate": False,
        "h_alpha_final_p": 1,
        "h_beta_i": 0.0,
        "h_beta_f": 0.0
    },
    "a3rl_5to5": {
        "offline_ratio": 0.5,
        "use_density": True,
        "heuristics": "adv",
        "start_a3rl": 100000,
        "use_alternate": True,
        "steps_a3rl": 5,
        "steps_rlpd": 5,
        "h_alpha_final_p": 1,
        "h_beta_i": 0.4,
        "h_beta_f": 1.0
    },
     "a3rl_5to5_noanneal": {
        "offline_ratio": 0.5,
        "use_density": True,
        "heuristics": "adv",
        "start_a3rl": 100000,
        "use_alternate": True,
        "steps_a3rl": 5,
        "steps_rlpd": 5,
        "h_alpha_final_p": 1,
        "h_beta_i": 0.0,
        "h_beta_f": 0.0
    },
    "rlpd": {
        "offline_ratio": 0.5,
        "use_density": False,
        "heuristics": "none",
        "start_a3rl": 300000,
        "use_alternate": False,
    },
    "a3rl_20to5": {
        "offline_ratio": 0.5,
        "use_density": True,
        "heuristics": "adv",
        "start_a3rl": 100000,
        "use_alternate": True,
        "steps_a3rl": 20,
        "steps_rlpd": 5,
        "h_alpha_final_p": 1,
        "h_beta_i": 0.4,
        "h_beta_f": 1.0
    },
}

# OLD RUN FORMAT:
# v8_halfcheetah_r_active_k4_a0.3_l0.1_s1001_321323_121212

# NEW RUN FORMAT:
# v05_halfcheetah_a3rl_interweave_a0.3_l0.1_s1000_250421_201702

def generate():
    currtime, printtime = get_time()
    time_folder = f"{RUNS_DIR}/{currtime}"
    if not os.path.exists(time_folder):
        os.makedirs(time_folder)
    # * GENERATE MODE
    n = len(list_env_name) * len(list_a_l) * len(list_seed)
    variant_info = VARIANTS[variant]
    offline_ratio = variant_info["offline_ratio"]
    heuristics = variant_info["heuristics"]
    use_density = variant_info["use_density"]
    start_a3rl = variant_info.get("start_a3rl", 300000)
    use_alternate = variant_info.get("use_alternate", False)
    steps_a3rl = variant_info.get("steps_a3rl", 0)
    steps_rlpd = variant_info.get("steps_rlpd", 0)
    h_beta_i = variant_info.get("h_beta_i", 1.0)
    h_beta_f = variant_info.get("h_beta_i", 1.0)
    h_alpha_final_p = variant_info.get("h_alpha_final_p", 1.0)
    for env in list_env_name:
        env_core, p_simple, p_medium, p_expert = env
        env_name = f"{env_core}_{p_simple}_{p_medium}_{p_expert}"
        for h_alpha, h_lambda in list_a_l:
            for seed in list_seed:
                project_name = f"a3rl1_{env_core}"
                group_name = f'{version}_{env_name}_{variant}'
                run_name = f'{group_name}_a{h_alpha}_l{h_lambda}_s{seed}_{currtime}'
                command = f"sbatch -p gpu -G1 -c1 -J {run_name} -d singleton -o /home-nfs/hung/slurm_out/%j.out ~/gen/scripts/a3rl/sbatch.sh {env_name} {project_name} {group_name} {offline_ratio} {h_alpha} {h_lambda} {heuristics} {use_density} {start_a3rl} {use_alternate} {steps_a3rl} {steps_rlpd} {p_simple} {p_medium} {p_expert} {h_beta_i} {h_beta_f} {h_alpha_final_p} {env_core} {seed} {run_name}\n"
        
                with open(f"{time_folder}/a3rl_{currtime}_{run_name}.sh", "w") as f: 
                    f.write(command)
                    f.close()
                with open(f"{time_folder}/_sheets_summary.txt", "a") as f:
                    f.write('\t'.join([printtime, env_name, version, variant, str(seed), str(h_alpha), str(h_lambda)]))
                    f.write('\n')
                    f.close()

    print(f"Running {n} runs. Time: {currtime}")
    with open(f"{time_folder}/_runs_summary.txt", "w") as f:
        f.write(f"variant: {variant}\n")
        f.write(f"list_a_l = {list_a_l}\n")
        f.write(f"list_seed = {list_seed}\n")
        f.write(f"list_env_name = {list_env_name}\n")
        f.write(f"Total tasks: {n}")
        f.close()

    with open(f"{time_folder}/all.sh", "w") as f:
        f.write(
f"""#!/bin/bash
for script in {time_folder}/a3rl_*.sh; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    fi
done""")
        f.close()

    for script in os.listdir(time_folder):
        if script.endswith(".sh"):
            script_path = os.path.join(time_folder, script)
            os.chmod(script_path, 0o755)
            
if __name__ == "__main__":
    generate()