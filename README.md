# Build the Library
``` bash
# Build
python setup_flashinfer.py build_ext --inplace
# Import (Currently fail)
python -c "import flashinfer_test._HIP"
# Only build flashinfer
hipcc -c flashinfer.cu -I./include_hip -o flashinfer.o 
```

# setup_flashinfer.py
The building code.

# include
`include` directory is the original code.
`include_hip` directory is `include` being hipified by `hipify-perl` and some other adjustment

* Difference
    * math.cuh
        ``` C++
        // include
        asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(y) : "f"(x), "r"(delta));
        // include_hip
        __shfl_xor_sync(0xffffffffffffffffULL, x, delta);
        ```
        Shuffle operations work differently between CUDA and HIP. I currently modify it to make it work.

# flashinfer
`flashinfer.cu` is the original code.
`flashinfer.hip` is `flashinfer.cu` being hipified by `hipify-perl` and some other adjustment

* Difference
    * Term `CUDA` to Term `HIP` (Shipped)
    * `max` adjustment
        ``` C++
        // flashinfer.cu
        constexpr size_t vec_size = std::max(
        static_cast<size_t>(16 / flashinfer::quant::size_of_type<DTypeIn>() /
                            FoldFactor),
        static_cast<size_t>(head_dim / 32));
        // flashinfer.hip
        constexpr size_t vec_size =
        ( (16 / flashinfer::quant::size_of_type<DTypeIn>() / FoldFactor)
            > (head_dim / 32)
        )
        ? (16 / flashinfer::quant::size_of_type<DTypeIn>() / FoldFactor)
        : (head_dim / 32);
        ```
        `max()` can't be seen as a constant in `HIP`, so we should use condition to set the paramenter.
        Similar for the `
        ``` C++
        // flashinfer.cu
        constexpr size_t vec_size =
        std::max(static_cast<size_t>(16 / flashinfer::quant::size_of_type<T>()),
                static_cast<size_t>(head_dim / 32));
        // flashinfer.hip
        constexpr size_t vec_size_a =
        static_cast<size_t>(16 / flashinfer::quant::size_of_type<T>());

        constexpr size_t vec_size_b =
            static_cast<size_t>(head_dim / 32);

        constexpr size_t vec_size = (vec_size_a > vec_size_b ? vec_size_a : vec_size_b);
        ```
        Other similar adjustment are skipped.