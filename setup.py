from setuptools import setup, find_packages

setup(
    name='gym_drone',
    version='0.0.1',  # Use semantic versioning: https://semver.org/
    author='Nikhl Melgiri',
    author_email='melgirinik@gmail.com',
    description='A Gymnasium environment for multi-agent drone simulation with guns and targets.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # If your README is in Markdown
    url='https://github.com/Engineering-Geek/gym-drone',  # Project home page or repository URL
    license='MIT',  # Or whatever license you choose
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[
        'gymnasium',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',  # Specify the Python versions your project supports
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
)
