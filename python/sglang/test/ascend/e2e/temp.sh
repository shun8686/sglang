cp -r /root/.cache/.cache/opencv /tmp
cd /tmp

# 层级1：最基础的配置/核心库（无任何依赖）
dpkg -i libdrm-common*.deb libsensors-config*.deb libx11-data*.deb
dpkg -i libxau6*.deb libxdmcp6*.deb

# 层级2：xcb基础库（依赖层级1）
dpkg -i libxcb1*.deb

# 层级3：X11核心库（依赖层级2）
dpkg -i libx11-6*.deb libx11-xcb1*.deb libxext6*.deb

# 层级4：xcb扩展库（依赖层级3）
dpkg -i libxcb-dri2-0*.deb libxcb-glx0*.deb libxcb-present0*.deb libxcb-randr0*.deb
dpkg -i libxcb-shm0*.deb libxcb-sync1*.deb libxcb-xfixes0*.deb libxfixes3*.deb
dpkg -i libxshmfence1*.deb libxxf86vm1*.deb libxcb-dri3-0*.deb

# 层级5：drm/llvm/传感器库（依赖层级1-4）
dpkg -i libdrm2*.deb libdrm-amdgpu1*.deb libdrm-nouveau2*.deb libdrm-radeon1*.deb
dpkg -i libllvm15*.deb libelf1*.deb libsensors5*.deb

# 层级6：Mesa基础库（依赖层级5）
dpkg -i libglapi-mesa*.deb libglvnd0*.deb libglx0*.deb

# 层级7：GL核心库（依赖层级6）
dpkg -i libgl1*.deb libglx-mesa0*.deb

# 层级8：最终主包（依赖层级7）
dpkg -i libgl1-mesa-dri*.deb libgl1-mesa-glx*.deb

#######################################
# 第三步：强制配置所有未完成的包（解决依赖闭环）
#######################################
dpkg --configure -a

#######################################
# 第四步：验证安装结果（确认所有包正常）
#######################################
echo -e "\n===== 最终安装状态验证 ====="
dpkg -l libgl1-mesa-dri libgl1-mesa-glx libgl1 libglx0 | grep -E "ii|Name"
