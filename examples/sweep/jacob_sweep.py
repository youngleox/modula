import os
import subprocess
import time
import fire 
import string
from pprint import pprint
from functools import partial


def get_free_gpus():
    """Returns a list of indices of GPUs that are currently not in use."""
    try:
        # Get the status of all GPUs
        gpu_status = subprocess.check_output("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits", shell=True)
        free_gpus = []
        for line in gpu_status.splitlines():
            gpu_index, memory_used = line.decode().split(',')
            if int(memory_used.strip()) == 0:  # Assuming a GPU with no memory used is free
                free_gpus.append(int(gpu_index.strip()))
        return free_gpus
    except subprocess.CalledProcessError as e:
        print("Failed to execute nvidia-smi")
        return []


def submit_job_to_gpu(cuda_script, gpu_index, debug=False):
    """Submits a CUDA job to a specific GPU."""
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    data_prefix = "IMAGENET_PATH=/var/datasets/imagenet100-256 " # hacked in
    cuda_prefix = f"CUDA_VISIBLE_DEVICES={gpu_index} "
    cuda_script = data_prefix + cuda_prefix + cuda_script.format(gpu=gpu_index)
    # Here, you can add code to run your CUDA script with the specified arguments.
    # This is a placeholder for the actual job submission.
    print(f"Running {cuda_script} on GPU {gpu_index}")

    # run job in background and suppress printing
    if debug:
        subprocess.Popen(cuda_script, shell=True) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(cuda_script, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return    


def generate_sweeps(script_template, **kwargs):

    sweeps = {
        'width':   ([32, 64, 128, 256, 512, 1024], 128),
        'depth':   ([2, 4, 8, 16, 32, 64], 3),
    }
    
    wd = 0.01

    if (kwargs['opt'] == 'adam') and (not kwargs['normalize']):
        lrs = [0.00012207031, 0.00024414062, 0.00048828125, 0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125]
        beta2 = 0.99
    else:
        lrs = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        beta2 = -1.0
       
    formattable_vars = set([x[1] for x in string.Formatter().parse(script_template) if x[1] is not None])
    format_mapping = {k: v for k, v in kwargs.items() if k in formattable_vars}
    
    base_tag = "_".join([f"{k}_{v}" for k, v in format_mapping.items()])
    
    for k, v in format_mapping.items():
        script_template = script_template.replace(f"{{{k}}}", f"{v}")
    script_template = script_template.replace("{beta2}", str(beta2))
    script_template = script_template.replace("{wd}", str(wd))

    all_scripts = []
        
    if kwargs['sweep'] == 'all':
        variables_to_sweep = sweeps.keys()
    else:
        variables_to_sweep = [kwargs['sweep']]
        
    for sweep_var in variables_to_sweep:        
        # get the default value except for the current sweep
        train_kwargs = {k: v[1] for k, v in sweeps.items()}
        
        for value in sweeps[sweep_var][0]:
            # get the sweep value and override the default
            train_kwargs[sweep_var] = value

            # sweep learning rates
            for lr in lrs:
                train_kwargs['lr'] = lr
                
                script = script_template
                for k, v in train_kwargs.items():
                    script = script.replace(f"{{{k}}}", f"{v}")
                    
                tag = base_tag + "_".join([""] + [f"{k}_{v}" for k, v in train_kwargs.items()])                
                script = script.replace("{tag}", tag)
                all_scripts.append(script)           

    return all_scripts


def run(**kwargs):
    
    script_template = \
        "python main.py " + \
        "--log_dir logs/{tag} " + \
        "--log_interval 100 " + \
        "--seed 0 " + \
        "--batch_size 128 " + \
        "--train_steps 10000 " + \
        "--test_steps 100 " + \
        "--dataset {dataset} " + \
        "--arch {arch} " + \
        "--depth {depth} " + \
        "--width {width} " + \
        "--context 128 " + \
        "--num_heads 8 " + \
        "--d_embed {width} " + \
        "--loss xent " + \
        "--lr {lr} " + \
        "--beta1 0.9 " + \
        "--beta2 {beta2} " + \
        "--wd {wd} "
    
    if kwargs['sn']:
        script_template += "--normalize "
    
    # script_template += \
    #     "1> logs/{tag}/out.log " + \
    #     "2> logs/{tag}/err.log"
    
    # ignore checks (add to here if adding more features)
    args_to_ignore = ['opt', 'sn', 'sweep', 'debug']
    
    # check if argument is compatible with the script template    
    formattable_vars = set([x[1] for x in string.Formatter().parse(script_template) if x[1] is not None])
    for k, w in kwargs.items():
        if k not in formattable_vars and k not in args_to_ignore:
            raise ValueError(f"Unknown argument: {k}")        
    
    scripts = generate_sweeps(script_template, **kwargs)
    scripts_run = 0
    pprint(scripts)
    print(f"Total number of scripts: {len(scripts)}")

    # Generate jobs with different arguments in a loop
    while len(scripts) > 0:
        free_gpus = get_free_gpus()

        if free_gpus:
            cuda_script = scripts.pop(0)
            scripts_run += 1

            submit_job_to_gpu(cuda_script, free_gpus[0], debug=kwargs['debug'])
            time.sleep(30)  # Adjust or remove sleep as needed
            print(f"[progress: {scripts_run}/{len(scripts)}]")
        else:
            time.sleep(30)  # Wait before checking again
    return
    


if __name__ == "__main__":
    """
    arch, opt_name, dataset, normalize, sweep_variable='all'
    
    python jacob_sweep.py --arch gpt --opt adam --dataset openwebtext --sn --sweep all
    python jacob_sweep.py --arch gpt --opt adam --dataset openwebtext --sn --sweep all
    """
        
    fire.Fire(run)
            
    
    # opt = 'adamw'
    # script_generator = generate_sweeps(script_template, arch, opt_name, dataset, normalize)
    # # get total number of scripts as a dry run
    # total_scripts = sum(1 for _ in script_generator)

    # script_generator = get_script(opt)
    # scripts_run = 0

    # # Generate jobs with different arguments in a loop
    # while True:
    #     free_gpus = get_free_gpus()

    #     if free_gpus:
    #         try:
    #             cuda_script = next(script_generator)
    #             scripts_run += 1
    #         except StopIteration:
    #             print('all scripts run')
    #             break
    #         submit_job_to_gpu(cuda_script, free_gpus[0])
    #         time.sleep(30)  # Adjust or remove sleep as needed
    #         print(f"[progress: {scripts_run}/{total_scripts}]")
    #     else:
    #         time.sleep(30)  # Wait before checking again