******************************************
Drawing 2D Julia Set Fractals using CUDA
******************************************

Dependencies
#############

Well, you just need CUDA toolkit to run this program.

Download CUDA toolkit `here <https://developer.nvidia.com/cuda-downloads>`_.

Make sure to follow all the installation steps mentioned in the `CUDA Documentation <https://docs.nvidia.com/cuda/>`_.

How to run
############

Just run the make file by executing the command: ``make``

The executable takes a single command line parameter.

``Usage: ./a.out <CPU or GPU>`` , enter GPU if you want to run it on GPU, or CPU if otherwise.


PARAMETERS
###########

- DIM: The dimension of the output image.
- MAXITER: Basically denotes the upper bound of the terms in the series.
- MAXMAGNITUDE: Denotes the upper bound on the magnitude for a point to be in Julia Set.
- EQUATION: Denotes the acutal equation.
- RED, GREEN, BLUE: all values between 0 to 255, for RGB value of a pixel.
- REALPARTCOM - Real part of the complex constant C.
- COMPLEXPARTCOM - Complex part of the complex constant C.
- SCALE - Scale of the image.
- FILENAME - output image filename.

Here's a example image:

.. image:: image.jpg
   :align: center
   :alt: Julia Set (C = -0.54 + 0.54i)

Note: the above image is 10000 x 10000 pixels, and it takes 22.6 microseconds for NVIDIA 940MX 4GB GPU to compute all values. 