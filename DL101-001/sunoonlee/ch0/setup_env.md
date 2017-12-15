### ch0 - 搭建环境：docker + tensorflow

系统：macOS Sierra 10.12.1

Tensorflow 是深度学习领域的流行开源库，主要功能是用 data flow graph 来进行数值计算。Tensorflow 的名称包含两部分：tensor 意为张量，flow 则如同张量在 computational graph 中传播的状态。

macOS 下安装 tensorflow 有四种方式：pip, virtualenv, Docker 和源码。其中 Docker 是近来流行的容器化平台，通过把运行环境打包成镜像（image），让软件与底层系统隔离，更便于移植。运行中的镜像实例叫做容器（container)。安装好 Docker 后需要让它在后台先运行起来，这样输入 docker 命令才能连接上。

Tensorflow 的 Docker 镜像无需专门下载，在第一次输入一个镜像的运行命令时 docker 会自动为你下载。tensorflow 的镜像同时发布在 gcr.io 和 dockerhub 上，镜像名的区别是后者没有 `gcr.io/` 前缀。

下载镜像的时候可能遇到网络问题，需要科学上网。我这边试了几次才成功下载好 dockerhub 上的镜像。gcr.io 的镜像似乎更难下载。

启动容器的命令是：
`docker run -it -p 8888:8888 tensorflow/tensorflow`

默认使用 Python 2。如果需要 Python 3 的话，可以在镜像名后注明相应的 [tag](https://hub.docker.com/r/tensorflow/tensorflow/tags/)，比如 `tensorflow/tensorflow:latest-py3`。另外，可以用 -v 来指定容器与本地的文件夹映射。

参考：
* [Installing TensorFlow on Mac OS X  |  TensorFlow](https://www.tensorflow.org/install/install_mac)

