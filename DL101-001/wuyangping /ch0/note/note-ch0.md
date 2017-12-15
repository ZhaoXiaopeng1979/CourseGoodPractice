#ch0 学习笔记

##Windows 10环境下Docker 安装
1. 安装Docker toolbox
2. 在Docker terminal或cmd窗口中
  Docker run -it tensorflow/tensorflow.
3. 第一次运行会下载docker image，需要花一些时间。
4. 下载完成后再次运行
  Docker run -it -p 8888:8888 tensorflow/tensorflow
  命令行会显示jupyter notebook的连接token
5. 用浏览器打开 http://192.168.99.100:8888 ，第一次连接需要输入token。
    Note：ip地址可以用docker-machine ls查看

##TensorBoard
1. 可以用6006端口连接到docker容器中的Tensorboard，以便查看TensorFlow的运行进度，使用以下命令.
    Docker run -it -p 8888:8888 -p 6006:6006 tensorflow/tensorflow
2. 然后用以下命令启动TensorBoard.
    Docker run tensorflow/tensorflow tensorboard --logdir ./
3. 在浏览器中打开http://192.168.99.100:6006, 即可连接到TensorBoard
