"""
Generates a dataset for training on a SLURM cluster.
Each route file is parallelized on its own machine.
Monitors the data collection and continues crashed processes.
Best run inside a tmux terminal.
"""

from datetime import datetime
import os
import subprocess
import time
import glob
import json
from pathlib import Path
import random
import re


def make_bash(
    code_dir, 
    route_file_number, 
    agent_name, 
    route_file, 
    ckeckpoint_endpoint, 
    save_pth, 
    seed, 
    carla_root, 
    town, 
    repetition
):
    """
    为一条 route XML 生成一个“启动 CARLA，然后运行 Leaderboard evaluator”的 Shell 脚本
    """

    # 从数据保存路径推导 SLURM 文件保存路径
    save_slurm = save_pth.replace("data/", "slurm/")

    # 生成启动脚本路径
    # 不是交给 sbatch 的 SLURM 脚本，而是实际启动 CARLA 的脚本
    jobfile = f"{save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh"
    # 创建脚本所在目录
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)

    # 构造 evaluator 命令
    run_command = \
        "python leaderboard/leaderboard/leaderboard_evaluator_local.py \
            --port=${FREE_WORLD_PORT} \
            --traffic-manager-port=${TM_PORT} \
            --traffic-manager-seed=${TM_SEED} \
            --routes=${ROUTES} \
            --repetitions=${REPETITIONS} \
            --track=${CHALLENGE_TRACK_CODENAME} \
            --checkpoint=${CHECKPOINT_ENDPOINT} \
            --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} \
            --debug=0 \
            --resume=${RESUME} \
            --timeout=600"

    # 构造启动脚本的完整文本
    qsub_template = f"""
        #!/bin/bash
        export SCENARIO_RUNNER_ROOT={code_dir}/scenario_runner_autopilot
        export LEADERBOARD_ROOT={code_dir}/leaderboard_autopilot

        # carla
        export CARLA_ROOT={carla_root}
        export CARLA_SERVER={carla_root}/CarlaUE4.sh
        export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI/carla
        export PYTHONPATH=$PYTHONPATH:leaderboard_autopilot
        export PYTHONPATH=$PYTHONPATH:scenario_runner_autopilot
        export REPETITIONS=1
        export DEBUG_CHALLENGE=0
        export TEAM_AGENT={agent_name}
        export CHALLENGE_TRACK_CODENAME=MAP
        export ROUTES={route_file}
        export TOWN={town}
        export REPETITION={repetition}
        export TM_SEED={seed}

        export CHECKPOINT_ENDPOINT={ckeckpoint_endpoint}
        export TEAM_CONFIG={route_file}
        export RESUME=1
        export DATAGEN=1
        export SAVE_PATH={save_pth}

        echo "Start python"

        export FREE_STREAMING_PORT=$1
        export FREE_WORLD_PORT=$2
        export TM_PORT=$3

        echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
        echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
        echo "TM_PORT: $TM_PORT"

        bash {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &

        sleep 180

        {run_command}    
        """

    # 创建启动脚本文件，并把模板内容写进去
    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile


def get_running_jobs(jobname, user_name):
    """
    从 squeue 中找出指定用户、指定名称的作业，并解析出数量、路线编号和 SLURM job ID
    """

    # 执行 Shell 查询并把结果按行拆分
    job_list = subprocess.check_output(
        (
            f"SQUEUE_FORMAT2='jobid:10,username:{len(username)},name:130' squeue --sort V | grep {user_name} | \
                grep {jobname} || true"
        ),
        shell=True,
    ).decode("utf-8").splitlines()

    currently_num_running_jobs = len(job_list)
    #  line is sth like "4767364   gwb791 eval_julian_4170_0   "
    routefile_number_list = [line.split("_")[-2] + "_" + line.split("_")[-1].strip() for line in job_list]
    pid_list = [line.split(" ")[0] for line in job_list]
    return currently_num_running_jobs, routefile_number_list, pid_list


def get_last_line_from_file(filepath): # this is used to check log files for errors
    """
    不从头读取整个大日志，而是直接移动到文件末尾，找到最后一行，用于查看错误
    """
    try:
        with open(filepath, "rb", encoding="utf-8") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
    except:
        last_line=""
    return last_line


def cancel_jobs_with_err_in_log(logroot, jobname, user_name):
    """
    查看当前每个采集作业的输出日志，如果最后一行出现已知崩溃信息，就取消该 SLURM 作业
    """

    # check if the log file contains certain error messages, then terminate the job
    print("Checking logs for errors...")
    _, routefile_number_list, pid_list = get_running_jobs(jobname, user_name)
    for i, rf_num in enumerate(routefile_number_list):
        # 构造当前路线的标准输出日志
        logfile_path = os.path.join(logroot, f"run_files/logs/qsub_out{rf_num}.log")
        # 读取日志最后一行
        last_line = get_last_line_from_file(logfile_path)
        # 默认不取消当前任务
        terminate = False
        if "Actor" in last_line and "not found!" in last_line:
            terminate = True
        if "Watchdog exception - Timeout" in last_line:
            terminate = True
        if "Engine crash handling finished; re-raising signal 11" in last_line:
            terminate = True
        if terminate:
            print(f"Terminating route {rf_num} with pid {pid_list[i]} due to error in logfile.")
            subprocess.check_output(f"scancel {pid_list[i]}", shell=True)


def wait_for_jobs_to_finish(logroot, jobname, user_name, max_n_parallel_jobs):
    """
    当作业数达到并发上限时，一直等待，直到至少空出一个提交位置
    """

    currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
    print(f"{currently_running_jobs}/{max_n_parallel_jobs} jobs are running...")
    # 循环计数器，用于控制日志错误检查频率
    counter = 0
    while currently_running_jobs >= max_n_parallel_jobs:
        if counter == 0:
            # 如果发现明显卡死的任务，取消它可能更快释放任务槽位
            cancel_jobs_with_err_in_log(logroot, jobname, user_name)
        time.sleep(5)
        currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
        counter = (counter + 1) % 4


def get_num_jobs(job_name, username):
    """
    它和 get_running_jobs() 都会查询 SLURM，但用途不同：
        get_running_jobs()：还需要路线编号和 job ID，用于监控
        get_num_jobs()：只需要作业数量和最大并发数，用于提交限流
    """

    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        ).decode('utf-8').replace('\n', ''))

    try:
        with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())
    except:
        max_num_parallel_jobs = 1

    return num_running_jobs, max_num_parallel_jobs


def get_which_partition(default):
    """
    从 partition.txt 动态选择 SLURM 分区，不可用时使用默认分区
    """
    try:
        with open('partition.txt', 'r', encoding='utf-8') as f:
            partition_name = f.read()
            if partition_name not in ["a100-galvani", "2080-galvani", "2080-preemptable-galvani", "a100-preemptable-galvani"]:
                partition_name = default
    except:
        print("partition.txt not found. Using default partition.")
        partition_name = default

    return partition_name


def make_jobsub_file(
    save_path_data, 
    jobname, 
    route_file_number, 
    partition_name, 
    repetition, 
    timeout="0-02:00"
):
    """
    生成真正交给 sbatch 的 SLURM 作业脚本
    """

    """
    创建三类目录和 SLURM 作业脚本路径
    run_files/
    ├── logs/        标准输出和错误日志
    ├── job_files/   交给 sbatch 的脚本
    ├── start_files/ 启动 CARLA 的脚本
    └── 2178_Rep0.sh SLURM 作业脚本路径
    """
    save_slurm = save_path_data.replace("data/", "slurm/")
    os.makedirs(f"{save_slurm}/run_files/logs", exist_ok=True)
    os.makedirs(f"{save_slurm}/run_files/job_files", exist_ok=True)
    os.makedirs(f"{save_slurm}/run_files/start_files", exist_ok=True)
    jobfile = f"{save_slurm}/run_files/job_files/{route_file_number}_Rep{repetition}.sh"

    # SLURM 模板
    qsub_template = f"""
        #!/bin/bash
        #SBATCH --job-name={jobname}_{route_file_number}
        #SBATCH --partition={partition_name}
        #SBATCH -o {save_slurm}/run_files/logs/qsub_out{route_file_number}.log
        #SBATCH -e {save_slurm}/run_files/logs/qsub_err{route_file_number}.log
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task=8
        #SBATCH --mem=40gb
        #SBATCH --time={timeout}
        #SBATCH --gres=gpu:1
        # -------------------------------

        echo "SLURMD_NODENAME: $SLURMD_NODENAME"
        echo "SLURM_JOB_ID: $SLURM_JOB_ID"
        echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
        scontrol show job $SLURM_JOB_ID

        dt=$(date '+%d/%m/%Y %H:%M:%S');
        echo "Job started: $dt"

        echo "Current branch:"
        git branch
        echo "Current commit:"
        git log -1
        echo "Current hash:"
        git rev-parse HEAD


        export FREE_STREAMING_PORT=`comm -23 <(seq 10000 10400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
        export FREE_WORLD_PORT=`comm -23 <(seq 20000 20400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
        export TM_PORT=`comm -23 <(seq 30000 30400 | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

        sleep 2

        echo "start python"
        pwd
        bash {save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh $FREE_STREAMING_PORT $FREE_WORLD_PORT $TM_PORT
        """

    # 将 SLURM 模板写入作业文件
    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile


if __name__ == "__main__":
    repetitions = 1                                         # 采集轮数
    repetition_start = 0                                    # 从第几轮开始

    default_partition = "YOUR_PARTITION"                    # SLURM 默认分区名
    job_name = "collect"                                    # SLURM 作业名称前缀
    username = "YOUR_USER"                                  # 提交 SLURM 作业的用户名

    code_root = r"/path/to/simlingo"                        # SimLingo 项目根目录的绝对路径
    carla_root = "/path/to/CARLA/root"                      # CARLA 安装根目录
    date = datetime.today().strftime("%Y_%m_%d")            # 获得当前日期，用于数据集命名
    dataset_name = "simlingo_v2_" + date                    # 生成数据集名称，例如：simlingo_v2_2026_07_22
    root_folder = r"database/"                              # 数据集相对于项目根目录的上层目录
    data_save_directory = root_folder + dataset_name        # 拼出相对数据集目录
    log_root = f"{data_save_directory}/slurm"               # 设置日志根目录

    route_folder = f"{code_root}/data/simlingo"             # 设置路线文件搜索根目录

    # find all .xml files in route_folder
    routes = glob.glob(f"{route_folder}/**/*balanced*/*.xml", recursive=True)
    routes_lb1 = glob.glob(f"{route_folder}/**/*lb1*/**/*.xml", recursive=True)

    routes = routes + routes_lb1                            # 将两组路线合并为一个列表

    # port_offset = 0 这个变量没被用过，注释掉
    job_number = 1                                          # 记录当前正在提交第几个任务
    meta_jobs = {}                                          # 保存本次运行提交的所有任务状态

    #shuffle routes
    random.seed(42)                                         # 固定 Python 随机种子
    random.shuffle(routes)                                  # 随机打乱路线顺序
    seed_counter = 1000000 * repetition_start - 1           # 初始化 Traffic Manager 的随机种子计数器

    num_routes = len(routes)                                # 统计一轮中共有多少个 XML 路线文件

    # 逐轮生成任务
    for repetition in range(repetition_start, repetitions):
        # 逐路线生成任务
        for route in routes:
            # 随机种子计数器
            seed_counter += 1

            # 确定路线对应的 Town
            try:
                town = re.search('Town(\\d+)', route).group(0)
            except:
                if 'validation' in route:
                    town = 'Town13'
                elif 'training' in route:
                    town = 'Town12'
                else:
                    print(f"Town not found in route {route}")
                    continue


            # ["simlingo", "validation_1_scenario", "routes_validation", "random_weather_seed_2_balanced_150"]
            # [数据集命名空间, 数据划分和路线复杂度, 原始路线来源, 路线生成方式]
            scenario_type = route.split("/")[-5:-1]
            # 重新拼接保留原相对目录结构
            scenario_type = "/".join(scenario_type)
            # 获取 XML 文件名
            routefile_number = route.split("/")[-1].split(".")[0]  # this is the number in the xml file name, e.g. 22_0.xml
            # 构造当前路线的结果和进度文件路径
            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"
            # 构造真正的数据集保存目录
            save_path = f"{code_root}/{data_save_directory}/data/{scenario_type}"
            Path(save_path).mkdir(parents=True, exist_ok=True)

            # 指定负责数据采集的 Agent 文件
            # evaluator 负责运行仿真
            # data agent 负责驾驶以及记录训练数据
            agent = f"{code_root}/team_code/data_agent.py"

            # 决定当前任务使用哪个 SLURM 分区
            partition_name = get_which_partition(default_partition)

            # 生成启动 CARLA 和 evaluator的 Shell 脚本
            bash_file = make_bash(
                code_root, 
                routefile_number, 
                agent, 
                route,
                ckpt_endpoint, 
                save_path, 
                seed_counter, 
                carla_root, 
                town, 
                repetition
            )

            # 生成 SLURM 作业脚本
            job_file = make_jobsub_file(
                save_path, 
                job_name, 
                routefile_number, 
                partition_name, 
                repetition, 
                "0-04:00"
            )

            # Wait until submitting new jobs that the #jobs are at below max
            # 当前用户相关 SLURM 作业数量，允许的最大作业数量
            num_running_jobs, max_num_parallel_jobs = get_num_jobs(
                job_name=job_name, 
                username=username
            )
            print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')

            # 如果当前作业数已经达到上限，就等待空位
            while num_running_jobs >= max_num_parallel_jobs:
                num_running_jobs, max_num_parallel_jobs = \
                    get_num_jobs(
                        job_name=job_name, 
                        username=username
                    )
                time.sleep(0.05)

            # 打印提交进度
            print(f"Submitting job {job_number}/{num_routes}: {job_name}_{routefile_number}. ", end="")
            time.sleep(1)

            # 获取 jobid
            jobid = subprocess.check_output(
                f"sbatch {job_file}", 
                shell=True
            ).decode("utf-8").strip().rsplit(" ", maxsplit=1)[-1]
            # 打印 SLURM 分配的 jobid
            print(f"Jobid: {jobid}")

            # 初始化当前路线元数据
            meta_jobs[jobid] = (
                False,              # 是否已完成
                job_file,           # SLURM 脚本
                ckpt_endpoint,      # 结果 JSON
                0                   # 重试次数
            )  # job_finished, job_file, result_file, resubmitted
            job_number += 1

    # 全部首次提交完毕后等待一秒
    time.sleep(1)

    # 循环结束标志
    training_finished = False
    while not training_finished:
        # 查询当前相关任务
        num_running_jobs, _, _ = get_running_jobs(job_name, username)
        # 打印当前任务数量
        print(f"{num_running_jobs} jobs are running... Job: {job_name}")
        # 检查运行任务的日志最后一行，如果发现已知错误，取消任务，以便后面重新提交
        cancel_jobs_with_err_in_log(log_root, job_name, username)
        time.sleep(20)

        # resubmit unfinished jobs
        # 遍历当前记录的所有 jobid
        for k in list(meta_jobs.keys()):
            job_finished, job_file, result_file, resubmitted = meta_jobs[k]
            need_to_resubmit = False
            # 只检查：尚未完成和重试次数小于3
            if not job_finished and resubmitted < 3:
                # 检查该 jobid 是否还在 SLURM 队列中
                # 如果结果为 0，说明任务已经离开队列，可能是：正常完成、超时、被取消、崩溃、SLURM 调度失败
                if int(subprocess.check_output(f"squeue | grep {k} | wc -l", shell=True).decode("utf-8").strip()) == 0:
                    # 先判断 result JSON 是否存在
                    if os.path.exists(result_file):
                        with open(result_file, "r", encoding="utf-8") as f_result:
                            evaluation_data = json.load(f_result)
                        # 读取 checkpoint 进度，[已完成数量, 总数量]
                        progress = evaluation_data["_checkpoint"]["progress"]
                        # progress 中不足两个数字或已完成数量小于总数量
                        if len(progress) < 2 or progress[0] < progress[1]:
                            # 没有完成，需要重试
                            need_to_resubmit = True
                        else:
                            # 遍历结果中的每条 route 记录
                            for record in evaluation_data["_checkpoint"]["records"]:
                                # 如果路线得分接近 0，认为采集失败，需要重试
                                if record["scores"]["score_route"] <= 0.00000000001:
                                    need_to_resubmit = True
                                # Agent 初始化失败，触发重试
                                if record["status"] == "Failed - Agent couldn\'t be set up":
                                    need_to_resubmit = True
                                # 普通失败，触发重试
                                if record["status"] == "Failed":
                                    need_to_resubmit = True
                                # 仿真崩溃，触发重试
                                if record["status"] == "Failed - Simulation crashed":
                                    need_to_resubmit = True
                                # Agent 崩溃，触发重试
                                if record["status"] == "Failed - Agent crashed":
                                    need_to_resubmit = True

                        # 任务完成
                        if not need_to_resubmit:
                            # delete old job
                            print(f"Finished job {job_file}")
                            meta_jobs[k] = (True, None, None, 0)

                    # 如果 result JSON 根本不存在，认为任务失败，需要重试
                    else:
                        need_to_resubmit = True

            # 重新提交失败任务
            if need_to_resubmit:
                # rename old error files to still access it
                # 从 SLURM 脚本路径中提取文件名，不带 .sh
                routefile_number = Path(job_file).stem
                # 打印即将重试的任务和旧 jobid
                print(f"Resubmit job {routefile_number} (previous id: {k}). Waiting for jobs to finish...")

                # 重新读取最大并发数量
                with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
                    max_num_parallel_jobs = int(f.read())
                # 如果当前作业数量达到并发上限，就等待出现空位
                wait_for_jobs_to_finish(log_root, job_name, username, max_num_parallel_jobs)

                # 获取当前 Unix 时间戳
                # 用于给旧日志备份目录命名
                # 避免不同重试日志互相覆盖
                time_now_log = time.time()
                # 创建旧日志备份目录
                os.system(f'mkdir -p "{log_root}/run_files/logs_{routefile_number}_{time_now_log}"')
                # 复制到刚创建的备份目录中
                os.system(f"cp {log_root}/run_files/logs/qsub_err{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}")
                os.system(f"cp {log_root}/run_files/logs/qsub_out{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}")

                # 再次提交同一个 SLURM 作业脚本，并取得新的 jobid
                jobid = subprocess.check_output(
                    f"sbatch {job_file}",
                    shell=True).decode("utf-8").strip().rsplit(" ", maxsplit=1)[-1]
                # 用新的 jobid 登记重试任务
                meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
                # 把旧 jobid 标记为已经处理完
                # 这里的 True 不代表旧任务运行成功
                meta_jobs[k] = (True, None, None, 0)
                # 打印新的 jobid
                print(f"resubmitted job {routefile_number}. (new id: {jobid})")

        # 一轮检查完成后等待 10 秒
        time.sleep(10)

        # 如果本轮开始时查询到的作业数为 0，就结束监控循环
        if num_running_jobs == 0:
            training_finished = True
