guide: https://pytorch.org/cppdocs/installing.html

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/Users/oscar/opt/libtorch ..
cmake --build . --config Release


or

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release