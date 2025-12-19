# LiFT-6DoF: Light Field based 6DoF Tracking of Previously Unobserved Objects

Project Page: https://nagonch.github.io/LiFT-6DoF/

Authors: [Nikolai Goncharov](https://www.linkedin.com/in/nikolai-goncharov-2931a31a5/), [James L. Gray](https://www.linkedin.com/in/james-gray-b6a250102/), [Donald G. Dansereau](https://www.linkedin.com/in/donald-dansereau/)

Paper link: https://arxiv.org/abs/2512.13007
Dataset link: https://ses.library.usyd.edu.au/handle/2123/34631

## 🛠️ Installation
Clone our repo recursively
```
git clone https://github.com/nagonch/LiFT-6DoF.git --recursive
```

Download SAM2 checkpoints
```
cd sam2/checkpoints && ./download_ckpts.sh 
cd ..
```

Build docker container
```
docker build -t lift6dof .
```

Run docker container
```
bash run_container.sh
```

## 🚀 Usage
Download test dataset sequence
```
curl -L \
  -H "User-Agent: Mozilla/5.0" \
  "https://ses.library.usyd.edu.au/bitstream/handle/2123/34631/jug_tilt_prod.zip?sequence=2&isAllowed=y" \
  -o jug_tilt_prod.zip
mkdir data
unzip jug_tilt_prod.zip -d data/jug_tilt_prod
```

Modify `config.yaml` as required and then run code
```
python run_experiment.py
```