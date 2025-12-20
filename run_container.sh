docker rm -f lift6dof

xhost +local:1000 && \
docker run \
    --name lift6dof \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$(pwd):$(pwd)" \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    --network=host \
    --ipc=host \
    -e DISPLAY="$DISPLAY" \
    -w "$(pwd)" \
    lift6dof:latest bash
