In this [example](./dotnetlibtorch.cs):
1. We create a DLPack tensor in C# ([`dotnetlibtorch.cs`](./dotnetlibtorch.cs))
2. Pass it to C function that uses libtorch to process the input ([`dotnetlibtorch.cpp`](./dotnetlibtorch.cpp))
3. Display the returned tensor in C# ([`dotnetlibtorch.cs`](./dotnetlibtorch.cs))

### Build instructions
```shell
# build C++ library with an exported C function that consumes a DLPack tensor and returns a DLPack tensor
# C++ library dotnetlibtorch.cpp adds 1 to the passed tensor
CMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.__path__[0])')
mkdir -p build
pushd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" ..
make
popd
```

### Running the example
```shell
# build and run C# caller
dotnet run dotnetlibtorch.cs

# dump export a linear PyTorch model in two versions: jit_scripted_model.pt and jit_traced_model.pt
python3 dotnetlibtorch.py
```

### Example output
```
Before passing to libtorch
type_code=kDLInt, bits=32, lanes=1, ndim=2, shape=[2,3], strides=[3,1]
(0, 0) = 0
(0, 1) = 1
(0, 2) = 2
(1, 0) = 3
(1, 1) = 4
(1, 2) = 5
After passing to libtorch
type_code=kDLInt, bits=32, lanes=1, ndim=2, shape=[2,3], strides=[3,1]
(0, 0) = 1
(0, 1) = 2
(0, 2) = 3
(1, 0) = 4
(1, 1) = 5
(1, 2) = 6
Called deleter
```
