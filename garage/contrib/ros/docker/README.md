# Sawyer Docker
## Instructions
- If you are using NVIDIA graphic cards
    install nvidia-docker first.
- change directory to garage root
    ```bash
    $ cd garage_root
    ```
- build
    ```bash
    # specify gpu if you are using NVIDIA graphic cards.
    $ ./garage/contrib/ros/docker/build.sh robot_name [cpu | gpu ]
    ```
- run
    ```bash
    # cpu
    $ ./garage/contrib/ros/docker/run.sh robot_name
    # gpu
    $ ./garage/contrib/ros/docker/run_gpu.sh robot_name
    ```
