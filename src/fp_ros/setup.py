from setuptools import find_packages, setup, find_namespace_packages
from glob import glob
import os

package_name = 'fp_ros'
submodules = "fp_ros/foundationpose"

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    package_data={
        'fp_ros': ['foundationpose/**/*'],
    },
    include_package_data=True,    
    zip_safe=False,  # Changed to False to ensure all files are extracted
    maintainer='dhruvstra',
    maintainer_email='dhruvsheth.linkit@gmail.com',
    description='ROS2 interface for foundationpose',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_pub = fp_ros.publish_data:main',
            'data_sub = fp_ros.fp_process:main',    
            'data_sub_multicam = fp_ros.fp_process_multicam:main',    

        ],
    },
)