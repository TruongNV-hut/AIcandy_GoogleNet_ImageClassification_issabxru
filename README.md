# GoogleNet and Image Classification

<p align="justify">
<strong>GoogleNet</strong> also known as Inception, is a deep convolutional neural network (CNN) architecture developed by Google. Introduced in 2014, it was a breakthrough in the field of computer vision, particularly in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), where it achieved top results. The architecture is known for its use of "Inception modules," which allow the network to capture features at multiple scales by using filters of different sizes simultaneously. This design enables the model to be both deep and computationally efficient, reducing the number of parameters while maintaining high accuracy in tasks like image classification and object detection.
</p>

## Image Classification
<p align="justify">
<strong>Image classification</strong> is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_GoogleNet_ImageClassification_issabxru.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_googlenet_train_odiiiyry.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_bretqhex/aicandy_model_pth_syliacip.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_googlenet_test_mucssnkn.py --image_path ../image_test.jpg --model_path aicandy_model_out_bretqhex/aicandy_model_pth_syliacip.pth --label_path label.txt
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_googlenet_convert_onnx_rtdytday.py --model_path aicandy_model_out_bretqhex/aicandy_model_pth_syliacip.pth --onnx_path aicandy_model_out_bretqhex/aicandy_model_onnx_bbcuqglm.onnx --num_classes 2
```

### More Information

To learn more about this project, [see here](https://aicandy.vn/ung-dung-mang-googlenet-vao-phan-loai-hinh-anh).

To learn more about knowledge and real-world projects on Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), visit the website [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




