echo "****************** Installing pytorch ******************"
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
pip install thop-0.0.31.post2005241907 -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom -i https://pypi.tuna.tsinghua.edu.cn/simple/


echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple/


echo ""
echo ""
echo "****************** Downgrade setuptools ******************"
pip install setuptools==59.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/


echo ""
echo ""
echo "****************** Installing wandb ******************"
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installing timm ******************"
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo ""
echo "****************** Installation complete! ******************"
