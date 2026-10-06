from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ur3_llm_control'


def files(pattern):
    return [f for f in glob(pattern) if os.path.isfile(f)]


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', files('launch/*.py')),
        ('share/' + package_name + '/config', files('config/*')),
        ('share/' + package_name + '/urdf', files('urdf/*')),
        ('share/' + package_name + '/prompts', files('prompts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thienhoang',
    maintainer_email='thienhoang@example.com',
    description='Điều khiển UR3 + gripper bằng ngôn ngữ tự nhiên qua LLM (MoveIt 2 + RViz).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_commander = ur3_llm_control.commander_node:main',
            'world_sim = ur3_llm_control.world_sim:main',
            'perception = ur3_llm_control.perception_node:main',
        ],
    },
)
