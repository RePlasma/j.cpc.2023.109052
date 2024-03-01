................................................
Ld

python3 ../my_roseNNa_examples/Ld/Ld.py

python3 modelParserONNX.py -w ../my_roseNNa_examples/Ld/Ld.onnx -f ../my_roseNNa_examples/Ld/Ld_weights.onnx

make library

gfortran -c ../my_roseNNa_examples/Ld/Ld.f90 -IobjFiles/
gfortran -o flibrary libcorelib.a Ld.o
./flibrary

................................................


L=1 d=10, t=   3.2900000000000003E-006  s
L=1 d=25, t=   5.0866666666666657E-006  s
L=1 d=50, t=   1.1923333333333335E-005  s
L=1 d=100, t=   3.3283333333333327E-005  s

L=5 d=10, t=   6.1066666666666678E-006  s
L=5 d=25, t=   1.3906666666666668E-005  s
L=5 d=50, t=   4.1303333333333328E-005  s
L=5 d=100, t=1.4385666666666668E-004  s

L=25 d=10, t=   2.0936666666666666E-005  s
L=25 d=25,   t=   5.9733333333333326E-005  s
L=25 d=50, t=   2.0209666666666664E-004  s
L=25 d=100, t=   7.0139666666666660E-004  s

L=50 d=10, t=   4.2963333333333322E-005  s
L=50 d=25, t=   1.2065333333333333E-004  s
L=50 d=50,   t=   4.0465666666666662E-004  s
L=50 d=100, t=   1.4926499999999999E-003  s