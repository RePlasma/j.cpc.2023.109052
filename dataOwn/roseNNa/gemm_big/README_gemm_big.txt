python3 ../goldenFiles/gemm_big/gemm_big.py

python3 modelParserONNX.py -w ../goldenFiles/gemm_big/gemm_big.onnx -f ../goldenFiles/gemm_big/gemm_big_weights.onnx

make library

gfortran -c ../examples/capiTester.f90 -IobjFiles/
gfortran -o flibrary libcorelib.a capiTester.o
./flibrary

python3 ../test/testChecker.py gemm_big