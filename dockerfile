FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


# Install Tools
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y ssh vim git iputils-ping net-tools tar unzip
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN echo "root:qwertyuiop" | chpasswd


# Install Anaconda
WORKDIR /root
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
RUN chmod +x Anaconda3-2024.06-1-Linux-x86_64.sh
RUN chmod +x anaconda_init.sh
RUN ./Anaconda3-2024.06-1-Linux-x86_64.sh -b -p /root/anaconda3
RUN ./anaconda_init.sh
RUN rm anaconda_init.sh
RUN rm Anaconda3-2024.06-1-Linux-x86_64.sh

# WorkDir
WORKDIR /root

# 启动命令
CMD service ssh start;sleep infinity