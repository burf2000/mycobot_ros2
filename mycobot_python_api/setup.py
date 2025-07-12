from setuptools import setup
import os
from glob import glob

package_name = 'mycobot_python_api'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='burf2000',
    maintainer_email='your@email.com',
    description='Python API to control MyCobot via MoveIt2 HTTP interface',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'moveit_http_server = mycobot_python_api.moveit_http_server:main',
            'simple_moveit_interface = mycobot_python_api.simple_moveit_interface:main',
        ],
    },
)