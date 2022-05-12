![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document

2022-05-11-ds-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for this workshop: [link](<url>)


## 👮Code of Conduct

Participants are expected to follow those guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

Raise your hand :hand:

## 🖥 Workshop website

* [Workshop](https://esciencecenter-digital-skills.github.io/2022-05-11-ds-gpu/)
* [Google Colab](https://colab.research.google.com)


## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Hanno Spreeuw

## 🧑‍🙋 Helpers

Aron Jansen, Suvayu Ali

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda
| Time  | Activity |
| ----- | -------- |
| 09:30 | Welcome |
| 09:45 | Introduction |
| 10:00 | Convolve an image with a kernel on a GPU using CuPy |
| 10:30 | **Coffee break** |
| 10:40 | Running CPU/GPU agnostic code using CuPy | 
| 11:30 | **Coffee break** |
| 11:40 | Run your Python code on a GPU using Numba |
| 12:30 | **Lunch break** |
| 13:30 | Introduction to CUDA |
| 14:30 | **Coffee break** |
| 14:40 | CUDA memories and their use |
| 15:30 | **Coffee break** |
| 15:40 | Data sharing and synchronization |
| 16:15 | Wrap-up |
| 16:30 | Drinks |

All times in the agenda are in the **CEST** timezone.

## 🔧 Exercises

### Challenge: fairer runtime comparison CPU vs. GPU

Compute again the speedup achieved using the GPU, but try to take also into account the time spent transferring the data to the GPU and back.

Hint: to copy a CuPy array back to the host (CPU), use the `cupy.asnumpy()` function.

Solution:
```python=
%timeit -n 100 convolve2d_gpu(cupy.asarray(input), cupy.asarray(gauss)).get()
# 29.6 ms ± 171 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
Significantly lower speedup factor (~90x)


cpu time: 2.68s, gpu time: 30.2ms, speedup:89
speedup factor: 91.06 GPU time: 29.1 ms

speedup_factor: 91.06
bye: speedup_factor: 104.8

### Challenge: compute prime numbers

Write a new function `find_all_primes_using_gpu` that uses `check_number_is_prime` instead of the inner loop of `find_all_primes`. How long does this new function take to find all primes up to 10000?



Solution:
```python=
def find_all_primes_using_gpu(number):
    all_primes = []
    result = np.zeros((1), np.int32)
    for i in range(2, number):
        check_number_is_prime[1, 1](i, result)
        if result[0] > 0:
            all_primes.append(result[0])
    return all_primes

%timeit find_all_primes_using_gpu(10_000)
# 5.06 s ± 8.25 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

1.38 ms ± 1.58 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
Wall time: 5.78 s

5.04 s ± 35.3 ms

5.49 s ± 106 ms

Solution: 
%timeit find_all_primes_using_gpu(10000)
4.97 s ± 10.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
25.6 ms ± 7.87 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

### Modify the raw CUDA kernel to work with variable block size and number of threads
Hint: use the CUDA special variables

Solution:
```c=
extern "C" __global__ void vector_add(const float * A, const float * B,
                                      float * C, const int size) {
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    C[item] = A[item] + B[item];
}

```

### Challenge: Compute prime numbers with CUDA

Given the following Python code, similar to what we have seen in the previous episode about Numba, write the missing CUDA kernel that computes all the prime numbers up to a certain upper bound.

```python
import numpy
import cupy
import math

# CPU
def all_primes_to(upper : int, prime_list : list):
    for num in range(2, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            prime_list[num] = 1

upper_bound = 100_000
all_primes_cpu = numpy.zeros(upper_bound, dtype=numpy.int32)
all_primes_cpu[0] = 1
all_primes_cpu[1] = 1
%timeit -n 1 -r 1 all_primes_to(upper_bound, all_primes_cpu)

# GPU
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
}
'''
# Allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)

# Compile and execute code
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)
%timeit -n 10 -r 1 all_primes_to_gpu(grid_size, block_size, (upper_bound, all_primes_gpu))

# Test
if numpy.allclose(all_primes_cpu, all_primes_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

There is no need to modify anything in the code, except writing the body of the CUDA `all_primes_to` inside the `check_prime_gpu_code` string, as we did in the examples so far.

Hint: look at the body of the Python `all_primes_to` function, and map the outermost loop to the CUDA grid.

Solution:
```python=
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int result = 1;
    
    if (i >= size) return;
    
    for (int j=2; j<i/2+1; j++) {
        if (i%j==0){
            result = 0;
            break;
        }
    }
    all_prime_numbers[i] = result;
}
'''
```
## 🧠 Collaborative Notes

### What is a GPU?
GPU stands for graphics processing unit, but it can also be used for general purpose computing (especially, highly parallel computing workloads).

CPUs and GPUs are structurally very different. CPUs are more general purpose, and in terms of die area, it's split between compute unit, cache, and controllers fairly evenly.  However for GPUs, most of the die area are compute units.  They are also designed with high throughput in mind.

The GPU is generally a separate device, and you have to explicitly move data to the *device* before you can perform any computation.

### Using Python for GPU programming
We will use image convolution as our example problem: for each pixel, we multiply all elements in the kernel, and repeat over the image. Some properties:
- numerically intensive
- highly parallel
- no conditionals

Q: Could we also use GPUs with Fourier transforms?
A: Yes

Let us create an example image to work with:
```python=
import numpy as np

input = np.zeros((2048, 2048))
input[8::16, 8::16] = 1

# plot
import pylab as pyl
%matplotlib inline # for notebooks

pyl.imshow(input[:64, :64])
```
![](https://i.imgur.com/PLpFLdB.png)


To create a kernel:
```python=
x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
distsq = x**2 + y**2
gauss = np.exp(-distsq)

# plot
pyl.imshow(gauss)
```
![](https://i.imgur.com/ydALops.png)

We can convolve the image using the CPU:
```python=
from scipy.signal import convolve2d as convolve2d_cpu
convolved_image_using_cpu = convolve2d_cpu(input, gauss)

# plot
pyl.imshow(convolved_image_using_cpu[:64, :64])
```
![](https://i.imgur.com/y9mQ7pI.png)

You can see after the convolution, the original image with a regular grid with points, we have a gaussians on the grid.  This is how a telescope image might look.

Let's do some performance measurements.
```python=
%timeit convolve2d_cpu(input, gauss)
# 2.82 s ± 44.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

*Aside:* the `%timeit` IPython magic accepts two options, `-n` to specify number of loops, and `-r` to specify the number of iterations in each loop.

Let's repeat this on a GPU.  However, first we need to copy our data & kernel to the GPU memory (as discussed in [What is a GPU?](https://hackmd.io/A2o_rDPvQo-ZAeRajo838A?both#What-is-a-GPU)).
```python=
from cupyx.scipy.signal import convolve2d as convolve2d_gpu

input_on_gpu = cupy.asarray(input) # copy data to GPU
gauss_on_gpu = cupy.asarray(gauss) # copy kernel to GPU

# convolve on the GPU
image_convolved_on_gpu = convolve2d_gpu(input_on_gpu, gauss_on_gpu)
```

Now to plot it, we can't do `pyl.imshow` as before, becase we need to copy our result from the GPU to main system memory.  So we can do something like this.

```python=
pyl.imshow(image_convolved_on_gpu.get()[:64, :64])
```
![](https://i.imgur.com/yj3ZfoJ.png)


Sometimes there are some hooks built-in that lets you compare or work with GPU data seamlessly; e.g. we can use the following to compare our GPU result with the CPU result we got earlier.
```python=
np.allclose(convolved_image_using_cpu, image_convolved_on_gpu)
# array(True)
```
We can do a performance comparison:
```python=
%timeit convolve2d_gpu(input_on_gpu, gauss_on_gpu)
# 17.3 ms ± 1.55 ms per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```
A fairer comparison that accounts for the data copy overhead shows a lower speedup.
```python=
%timeit -n 100 convolve2d_gpu(cupy.asarray(input), cupy.asarray(gauss)).get()
# 29.6 ms ± 171 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
This means, it is not efficient to send small jobs to the GPU. It is better to send data in large batches, and do large computations.

#### Some `numpy` tricks
Using numpy, we can sometime write hardware agnostic code (as we saw earlier with `np.allclose`).
```python=
from numpy import convolve as convolve1d

input1d = np.ravel(input)
gauss1d = np.ravel(gauss)

# alternate way to reshape to 1D
input1d = np.reshape(input, -1)

oned_convolved_array = convolve1d(input1d, gauss1d)

# no error!
oned_convolved_array_on_gpu = convolve1d(cupy.asarray(input1d), cupy.asarray(gauss1d))
```
So we can sometimes write convenience functions, that can work with arrays on the GPU transparently (e.g. a plotting function).

More information on CuPy/NumPy interoperability can be found in CuPy [docs](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#numpy).

### How do we speedup custom code?
Let us find some prime numbers. 
```python=
import numba as nb

# pure python implementation
def find_all_primes(upper):
    all_primes = []
    for i in range(2, upper):
        for j in range(2, i//2+1):
            if i%j == 0:
                break
        else:
            all_primes.append(i)
    return all_primes

%timeit find_all_primes(10_000)
# 185 ms ± 535 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
Let's speedup by JIT (just-in-time) compiling our python function:
```python=
# typically used as a function decorator
find_all_primes_jit_compiled = nb.jit(find_all_primes)

%timeit -n 10 find_all_primes(10_000)
# 18.2 ms ± 593 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

#### Using numba to write a GPU kernel
```python=
from numba import cuda

@cuda.jit
def check_number_is_prime(number, result):
    result[0] = 0
    for j in range(2, number//2+1):
        if number%j == 0:
            break
    else:
        result[0] = number
```
Note that the GPU cannot return a value.  We need to provide a memory location, where the GPU will write the result.  This is because the GPU is a separate device with it's own memory, and we do not have direct access to it.  We do this by passing the `result` array, which is set to *zero*, and the result is written to it only if the kernel finds a prime number.

```python=    
result = np.zeros((1), np.int32)
check_number_is_prime[1, 1](10, result)
result
# array([0], dtype=int32)
```
*Notes:*
- this will give a warning about under utilising the GPU, but we can ignore it at this point.
- the `[1, 1]` after the GPU kernel is to specify how many threads & blocks of memory the kernel will need.  A GPU always needs this information before it can be executed.

We can repeat the find primes exercise using the GPU kernel
```python=
def find_all_primes_using_gpu(number):
    all_primes = []
    result = np.zeros((1), np.int32)
    for i in range(2, number):
        check_number_is_prime[1, 1](i, result)
        if result[0] > 0:
            all_primes.append(result[0])
    return all_primes

%timeit find_all_primes_using_gpu(10_000)
# 5.06 s ± 8.25 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
However, now it is infact slower!  This is because we are asking the GPU repeatedly for results, and the data copy overhead adds up (as mentioned in the warning from cupy).

Let's use the `numba.vectorize` decorator to apply a function on arrays instead of single numbers.  We can use this to increase the workload on the GPU kernel later.

```python=
import math

from numba import vectorize

@vectorize
def discriminant(a, b, c):
    return math.sqrt(b**2 - 4*a*c)

discriminant(np.arange(2, 8), np.arange(3, 9), np.arange(-10, -4))
# array([ 9.43398113, 11.13552873, 12.36931688, 13.26649916, 13.89244399, 14.28285686])
```
##### Vectorising a GPU kernel
We can now vectorise our prime finding GPU kernel, and as a side benefit, now we can also return a result!

```python=
from numba import int32

@vectorize([int32(int32)], target="cuda")
def check_if_numbers_are_prime(number):
    for j in range(2, number//2+1):
        if number%j == 0:
            return 0  # note: break replaced by return statement
    else:
        return number
    
%timeit check_if_numbers_are_prime(np.arange(2, 10_000, dtype=np.int32))
# 3.68 ms ± 21.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
### Writing a raw (CUDA C) GPU kernel using CuPy
Let's consider the following Python function:
```python=
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
    return C
```

Equivalent CUDA C code would be:
```c=
extern "C" __global__ void vector_add(const float * A, const float * B, float C, const int size) {
    int item = threadId.x;
    C[item] = A[item] + B[item];
}
```
The CUDA library takes care of looping over all the items of the array thanks to the special variable `threadId`.

In CuPy this can be implemented as:
```python=
import cupy

size = 1024

a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

vector_add_code = r'''
extern "C" __global__ void vector_add(const float * A, const float * B,
                                      float * C, const int size) {
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''

vector_add_gpu = cupy.RawKernel(vector_add_code, "vector_add")
# execute: 1st argument - blocks, 2nd argument - threads
vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```
GPU always does indexing in 3-dimensions, so when executing the kernel we need to always specify all dimensions.

We can compare the results from the GPU with the CPU:
```python=
a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

np.allclose(c_cpu, c_gpu)
```
If we repeat the above computation, but now change the thread and block arguments:
```python=
vector_add_gpu((2, 1, 1), (size//2, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```
Now the computation happens in with half as many threads, with 2 blocks.  However, you will see the result does not match with the CPU computation.  That is because in our kernel, we were using `threadIdx.x` to index our array, with half as many threads, now only the first half of the array is being computed.  So the lesson here is that, we need to be mindful about the special variables we use, and how they behave when we change block size or number of threads.

CUDA special variables:
| Keyword |	Description |
|---------|-------------|
| threadIdx | 	the ID of a thread in a block |
| blockDim |	the size of a block, i.e. the number of threads per dimension
| blockIdx |	the ID of a block in the grid
|gridDim |	the size of the grid, i.e. the number of blocks per dimension

We can make the kernel work with variable block size and number of threads by adjusting how we index the array in our kernel.
```c=
extern "C" __global__ void vector_add(const float * A, const float * B,
                                      float * C, const int size) {
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    C[item] = A[item] + B[item];
}
```
Note that the index still covers all elements in the array, but now it does the correct calculation even for other block sizes. 

We can generalise the block size calculation with something like this:
```python=
size = 1024 # can be anything

threads_per_block = 1024
grid_size = (int(math.ceil(size/threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
```

To protect from writing to memory outside the bounds of our array, we can modify the kernel like this:
```c=
extern "C" __global__ void vector_add(const float * A, const float * B,
                                      float * C, const int size) {
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (item < size)
        C[item] = A[item] + B[item];
}
```

See the [prime number exercise](https://hackmd.io/A2o_rDPvQo-ZAeRajo838A?both#Challenge-Compute-prime-numbers-with-CUDA) for another example of indexing in CUDA kernels.

#### Discussion on memory on the GPU
There are different kinds on memory on the GPU.  *Global memory* is the largest block, and managed by the host.  Since it is accessible by both host and device, it is used for exchanging data.  Any thread local variables used within a kernel is stored in *registers*, which are typically very small in size.  The kernel also has access to a *local memory*, however it is much slower compared to registers.  There is also a block of *shared memory* that is accesible to all threads on the GPU.

#### Making a histogram on the GPU using shared memory
- Need to use `__syncthreads()` to make threads wait for other threads, so that the histogram is initialized fully before we continue.
- Need to use atomicAdd to avoid different threads modifying the same histogram entry at the same time, which would give a wrong result.

```python=
import math
import cupy
import numpy

size = 2**25

input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
output_gpu = cupy.zeros((size), dtype=cupy.int32)

histogram_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
    
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();
    
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();
    
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

histogram_gpu = cupy.RawKernel(histogram_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))
```

```python=
import pylab

pylab.hist(output_gpu.get())
```
![](https://i.imgur.com/uiTMyNH.png)


## 📚 Resources

* [Upcoming eScience Center workshops](https://www.esciencecenter.nl/digital-skills/)
* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [JupyterHub Guide](https://servicedesk.surfsara.nl/wiki/display/WIKI/User+Manual+-+Student)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [Fellowship Programme](https://www.esciencecenter.nl/fellowship-programme/)
* [eScience Center website](https://www.esciencecenter.nl/)
