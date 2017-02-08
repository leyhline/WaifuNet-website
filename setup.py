from setuptools import setup

setup(
    name='waifunet',
    packages=['waifunet'],
    include_package_data=True,
    install_requires=[
        'flask', 'keras', 'numpy', 'matplotlib', 'opencv-python'
    ]
)
