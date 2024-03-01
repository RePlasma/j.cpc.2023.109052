................................................
L30d30

python3 ../my_roseNNa_examples/L30d30/L30d30.py

python3 modelParserONNX.py -w ../my_roseNNa_examples/L30d30/L30d30.onnx -f ../my_roseNNa_examples/L30d30/L30d30_weights.onnx

make library

gfortran -c ../my_roseNNa_examples/L30d30/L30d30.f90 -IobjFiles/
gfortran -o flibrary libcorelib.a L30d30.o
./flibrary