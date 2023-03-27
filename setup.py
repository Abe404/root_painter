from setuptools import setup
from pathlib import Path
import io

current_dir = Path(__file__).parent
long_description = io.open(current_dir / "README.md", mode="r", encoding="utf-8").read()

setup(
  name = 'root_painter_trainer',
  package_dir = {'root_painter_trainer': 'trainer'},
  packages = ['root_painter_trainer'],
  version = '0.2.25.3',
  license = 'GPL-2.0', 
  description = 'Trainer (server component) for RootPainter',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Abraham George Smith',
  url = 'https://github.com/Abe404/root_painter',
  entry_points={
    'console_scripts': [
      'start-trainer = root_painter_trainer:start',
    ]
  },
  install_requires=[
    "scikit-image==0.19.3",
    "numpy==1.24.2",
    "scipy==1.10.0",
    "Pillow==9.3.0",
    "imagecodecs==2021.8.26",
    "torch==1.13.1",
    "torchvision==0.14.1"
  ],
  classifiers=[
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
    'Operating System :: OS Independent'
  ]
)
