gemm_small standard (in /examples with capiTester.f90)

python3 ../goldenFiles/gemm_small/gemm_small.py

python3 modelParserONNX.py -w ../goldenFiles/gemm_small/gemm_small.onnx -f ../goldenFiles/gemm_small/gemm_small_weights.onnx

make library

gfortran -c ../examples/capiTester.f90 -IobjFiles/
gfortran -o flibrary libcorelib.a capiTester.o
./flibrary

python3 ../test/testChecker.py gemm_small

................................................
ex_gemm_small_simple

python3 ../my_roseNNa_examples/gemm_small/gemm_small.py

python3 modelParserONNX.py -w ../my_roseNNa_examples/gemm_small/gemm_small.onnx -f ../my_roseNNa_examples/gemm_small/gemm_small_weights.onnx

make library

gfortran -c ../my_roseNNa_examples/gemm_small/ex_gemm_small_simple.f90 -IobjFiles/
gfortran -o flibrary libcorelib.a ex_gemm_small_simple.o
./flibrary

................................................

ex_gemm_small_batch (similar to ex_gemm_small_simple)

gfortran -c ../my_roseNNa_examples/gemm_small/ex_gemm_small_batch.f90 -IobjFiles/
gfortran -o flibrary libcorelib.a ex_gemm_small_batch.o
./flibrary