# LiFT-6DoF: Light Field based 6DoF Tracking of Previously Unobserved Objects

Project Page: https://nagonch.github.io/LiFT-6DoF/

Authors: [Nikolai Goncharov](https://www.linkedin.com/in/nikolai-goncharov-2931a31a5/), [James L. Gray](https://www.linkedin.com/in/james-gray-b6a250102/), [Donald G. Dansereau](https://www.linkedin.com/in/donald-dansereau/)

Paper link: https://arxiv.org/abs/2512.13007

## Installation

Install SAM2
```
cd sam2/checkpoints && ./download_ckpts.sh 
cd ..
pip install -e .
cd ..
```

Install Video Depth Anything
```
cd Video-Depth-Anything
pip install -r requirements.txt
cd ..
```