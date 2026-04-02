from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

description = """SMAC - StarCraft Multi-Agent Challenge

SMAC offers a diverse set of decentralised micromanagement challenges based on
StarCraft II game. In these challenges, each of the units of interest is
controlled by an independent, learning agent that has to act based only on
local observations, while the opponent's units are controlled by the built-in
StarCraft II AI.

The environment requires StarCraft II to be installed."""

setup(
    name='SMAC',
    version='0.1.0b1',
    description='SMAC - StarCraft Multi-Agent Challenge.',
    long_description=description,
    author='WhiRL',
    author_email='mikayel@samvelyan.com',
    license='MIT License',
    keywords='StarCraft, Multi-Agent Reinforcement Learning',
    url='https://github.com/oxwhirl/smac',
    packages=find_packages(exclude=["bin", "contrib", "docs", "tests"]),
    install_requires=[
        # --- 核心依赖更新 ---
        'pysc2>=3.0.0',
        's2clientprotocol>=4.10.1.75800.0',
        'absl-py>=0.1.0',
        
        # --- M4 Mac 关键修改 ---
        # 允许新版 Numpy 以适配 Apple Silicon，但需配合代码 Patch
        'numpy>=1.21',  
        
        # --- 解决依赖冲突 ---
        # 1. 放宽 Gym 版本，但锁定在 0.21 之前。
        #    警告：绝对不要升级到 gym>=0.26，否则 step() 函数会报错！
        "gym>=0.16.0, <0.21.0", 
        
        # 2. 删除了 'pyglet' 的硬性要求。
        #    让 Gym 自己去决定它需要哪个版本的 pyglet，从而解决 conflict。
        
        # 3. 限制 Protobuf 版本。
        #    新版(4.x/5.x)会导致 "Descriptors cannot not be created directly" 错误。
        "protobuf<=3.20.3",
    ],
    package_data={'smac.env.lbforaging.foraging': ['icons/*.png']}
)
