将四个gpio口设置为输出模式
sudo busybox devmem 0x02430070 w 0x08
sudo busybox devmem 0x02434040 w 0x04
sudo busybox devmem 0x02430068 w 0x08
sudo busybox devmem 0x02434080 w 0x05
开启最大功率
sudo nvpmodel -m 2
sudo jetson_clocks
设置用户权限，允许当前系统用户访问和使用Jetson.GPIO库


设置主板型号，目前jetpack6.1不会提前设置主板型号，每次控制GPIO时需要在终端设置主板型号：
export JETSON_MODEL_NAME=JETSON_ORIN_NANO
