cp -r /root/.cache/.cache/opencv /tmp
cd /tmp/opencv

echo "\n====== [Begin] 临时规避：安装OpenGL ======"
ls *.deb | xargs dpkg -i

dpkg --configure -a

echo -e "\n====== [End] 临时规避：安装OpenGL ======"
dpkg -l libgl1-mesa-dri libgl1-mesa-glx libgl1 libglx0 | grep -E "ii|Name"
