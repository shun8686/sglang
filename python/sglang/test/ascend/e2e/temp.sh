libgl_pkgs=$(apt list | grep libgl1-mesa)

if [ -z "$libgl_pkgs" ];then
  # 如有网, 可在线安装：
  # apt update
  # apt install libgl1-mesa-glx libgl1-mesa-dri -y
  # 如无网, 可离线安装：
  echo "\n====== [Begin] 临时规避：安装OpenGL ======"
  cp -r /root/.cache/.cache/opencv /tmp
  cd /tmp/opencv
  ls *.deb | xargs dpkg -i
  dpkg --configure -a
  dpkg -l libgl1-mesa-dri libgl1-mesa-glx libgl1 libglx0 | grep -E "ii|Name"
  echo -e "\n====== [End] 临时规避：安装OpenGL ======"
fi
