import os
from setuptools import find_packages, setup

package_name = 'anomaly_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', 'python3.10', 'site-packages', 'models'), [
            'models/weights.onnx',
            'models/helmet_best.pt',
            'models/smoking_best.pt',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='Integrated anomaly detector: fire/smoke, safety helmet, smoking, license plate, face recognition',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'anomaly_detector = anomaly_detector.anomaly_detector_node:main',
            'detect_lb = anomaly_detector.detect_lb:main',
            'detect_warn = anomaly_detector.detect_warn:main',
            'detect_warn1 = anomaly_detector.detect_warn1:main',
            'detect_warn2 = anomaly_detector.detect_warn2:main',
            'ptz_detect_warn = anomaly_detector.ptz_detect_warn:main',
            'new_ptz_detect_warn = anomaly_detector.new_ptz_detect_warn:main'
        ],
    },
)
