#CP2102 串口号0006 设置别名为wheeltec_mmwave_radar
echo  'KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE:="0777", GROUP:="dialout", SYMLINK+="wheeltec_mmwave_radar"' >/etc/udev/rules.d/wheeltec_mmwave_radar.rules
#CH9102，同时系统安装了对应驱动 串口号0006 设置别名为wheeltec_mmwave_radar
echo  'KERNEL=="ttyCH343USB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d4", MODE:="0777", GROUP:="dialout", SYMLINK+="wheeltec_mmwave_radar"' >/etc/udev/rules.d/wheeltec_mmwave_radar2.rules
#CH9102，同时系统没有安装对应驱动 串口号0006 设置别名为wheeltec_mmwave_radar
echo  'KERNEL=="ttyACM*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d4", MODE:="0777", GROUP:="dialout", SYMLINK+="wheeltec_mmwave_radar"' >/etc/udev/rules.d/wheeltec_mmwave_radar3.rules

service udev reload
sleep 2
service udev restart


