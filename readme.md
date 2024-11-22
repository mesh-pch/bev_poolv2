# BEV PoolV2 TensorRT Plugin

This repository provides a TensorRT plugin implementation for BEV PoolV2. The original custom operator only supports float data types. This repository extends the functionality to support both half-precision (FP16) and int8 data types.

## Features

- **Float (FP32) Support**: Original implementation.
- **Half-Precision (FP16) Support**: Added support for half-precision data types.
- **Int8 Support**: Added support for int8 data types.

## Installation

To install the plugin, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/mesh-pch/bev_poolv2.git
    cd bev-poolv2-tensorrt-plugin
    ```

2. Install GCC 7+ dependencies:
    ```bash
    # Add repository if ubuntu < 18.04
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install gcc-7
    sudo apt-get install g++-7
    ```
3. Modify the `source_env.bash` file to set the paths for TensorRT and cuDNN:
    ```bash
    export TENSORRT_DIR=/path/to/tensorrt
    export CUDNN_DIR=/path/to/cudnn
    ```

4. Source the environment variables:
    ```bash
    source source_env.bash
        ```
5. Build the plugin:
    ```bash
    mkdir build
    cd build
    cmake -DCMAKE_CXX_COMPILER=g++-7 -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} ..
    make -j$(nproc)
    ```

6. Install the plugin:
    ```bash
    sudo make install
    ```

## Usage

To use the BEV PoolV2 TensorRT plugin in your project, you can load the plugin using the following Python function:

```python
import os
import ctypes

def load_tensorrt_plugin() -> bool:
    lib_path = 'your_path/mmdeploy/lib/libmmdeploy_tensorrt_ops.so'
    success = False
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        success = True
    else:
        print(f'Could not load the library of TensorRT plugins. '
              f'Because the file does not exist: {lib_path}')
    return success
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This repository is based on the original BEV PoolV2 implementation. Special thanks to the original authors for their work.
