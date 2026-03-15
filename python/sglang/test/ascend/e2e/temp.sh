cp -r /root/.cache/.cache/opencv /tmp
cd /tmp/opencv

echo "\n======Begin to install opencv======"
ls *.deb | xargs dpkg -i

dpkg --configure -a

echo -e "\n===== 最终安装状态验证 ====="
dpkg -l libgl1-mesa-dri libgl1-mesa-glx libgl1 libglx0 | grep -E "ii|Name"
