same as ../libtorch, but now to see the scaling with the number of enties in input (batch size)


mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release
./inferenceTime_libtorch_batchsize