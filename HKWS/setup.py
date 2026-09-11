from setuptools import find_packages, setup

package_name = 'HKWS'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hksdk = HKWS.hksdk:main',
            'web = HKWS.web:main',
            'alarm = HKWS.alarm:main',
            'alarm_smart_fixed = HKWS.alarm_smart_fixed:main',
            'alarm_smart_diagnostic = HKWS.alarm_smart_diagnostic:main',
            'hikvision_camera_alarm = HKWS.hikvision_camera_alarm:main',
            'new_hksdk = HKWS.new_hksdk:main',
            'new_hksdk_1 = HKWS.new_hksdk_1:main'
        ],
    },
)
