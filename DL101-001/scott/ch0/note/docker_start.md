本周任务主要是搭建 Tensorflow 环境，虽说我用的是 Debian，直接 pip 安装 Tensorflow 更方便，不过既然是作业要求，便也安下 Docker ，算是做个复习--之前有搭过 Docker 版的 Jupyter Nbviewer。另外，也写个 Cheat Sheet，方便今后复习 Docker 命令。

## 安装 Docker 和 Tensorflow

Docker 安装非常方便，Debian 参考 [Get Docker for Debian or Raspbian - Docker](https://docs.docker.com/engine/installation/linux/debian/#/prerequisites) 安装即可，另外 Linux 内核版本需注意下，只支持 3.10 以上的内核版本，我的 Jessie 是由 Wheezy 升级而来，查看版本后，是 `3.2`，原因是升级系统时却没升级内核，谷歌下，运行 `sudo apt upgrade` 即把内核升级到了 `3.16`。

## Docker 基本操作

Docker 分为镜像和容器两部分，如果把镜像理解成 Python 里面的类，那容器就是实例。


### 镜像相关操作

拉取镜像：

`docker pull 仓库名:标签`

如：

```
$ docker pull ubuntu:14.04
```

列出镜像：

```
$ docker images
REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
nginx                   latest              6b914bbcb89e        39 hours ago        182 MB
ubuntu                  latest              0ef2e08ed3fa        2 days ago          130 MB
tensorflow/tensorflow   latest              ea40dcc45724        2 weeks ago         1.03 GB
```

列出部分镜像：

```
$ docker images ubuntu
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu              latest              0ef2e08ed3fa        2 days ago          130 MB
```

删除镜像：

`docker rmi [OPTION] IAMGE...` IMAGE 可以是镜像名、长 id、短 id 等。


### 容器相关操作

列出在运行容器：

```
$ docker container ls
```

列出所有运行过的容器：

```
$ docker container ls -a
```

运行一个容器：

```
$ docker run -i -t ubuntu /bin/bash
```

`run` 意味着首先在本地搜索 `ubuntu` 镜像，没有则在官方下载，`-i` 保证容器的 STDIN 是开启的，`-t` 创建伪 tty 终端，方便创建之后的交互，`/bin/bash` 是创建之后运行的命令，另外 `run` 之后若是接 `--name` 参数可以指定容器名称。

`docker container stop CONTAINER` 停止一个容器的运行，跟 IMAGE 一样，CONTAINER 也可以是短 id，长 id，或者名字。如：

```
$ docker container stop webserver
```

同理，`stop` 换成 `start` 之后可重新运行容器，换成 `rm` 即删除容器。

终止所有在运行的容器：

```
$ docker rm $(docker ps -a -q)
```

本地目录映射到 docker 容器：

```
sudo docker run -p 8888:8888 -v ~/Documents/GitRepoes/DeepLearning101:/notebooks/DeepLearning101 tensorflow/tensorflow
```


## Docker 小结

对于依赖简单，无状态的服务，用 Docker 做起来很方便，比如搭建一个自己 Jupyter Nbviwer，用来放一些私密的 ipynb 文档，Docker 一键就搞定了；而有些配置多，用 Docker 反而更麻烦，比如 Tensorflow，若我想换 python3，就得重新拉个镜像了，所以综合考虑，我还是选择用 pyenv 管理虚拟环境。




