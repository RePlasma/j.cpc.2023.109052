# Stand-alone roseNNa inference

Guide by [Bertrand Martinez](https://github.com/bertrandmartinez) and [Óscar Amaro](https://github.com/OsAmaro) (Jan 2024)

### Simple test, first run

Follow installation steps as described in https://github.com/comp-physics/roseNNa

Assuming roseNNa is installed at ```Users/user/opt/roseNNa```.

Before following these steps, remove ```capiTester.o``` , ```flibrary``` , and ```test.txt``` to make sure RoseNNa is working as expected.

Creates ```capiTester.o``` file
``` bash
gfortran -c capiTester.f90 -I/Users/user/opt/roseNNa/fLibrary/objFiles/
```


Creates ```flibrary```
``` bash
gfortran -o flibrary /Users/user/opt/roseNNa/fLibrary/libcorelib.a capiTester.o
```


Reads ```onnxModel.txt``` and ```onnxWeights.txt```. If these are not in the directory or are empty, empty files will be created
``` bash
./flibrary
```

You should see the output
``` bash
0.31045650796579061        0.0000000000000000        0.0000000000000000 
```


### Compiling your own models

