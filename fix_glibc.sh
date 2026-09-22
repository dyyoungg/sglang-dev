# 1. 找到系统上有 GLIBCXX_3.4.30 的 libstdc++
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30

# 2. 把它拷贝到 conda 环境的 lib 目录
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /root/miniconda3/envs/llava/lib/

# 3. 重新建立软链接
cd /root/miniconda3/envs/llava/lib/
ln -sf libstdc++.so.6.0.30 libstdc++.so.6

# 4. 验证
strings /root/miniconda3/envs/llava/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30