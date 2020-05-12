#ifndef COMPLEX
#define COMPLEX

class Complex {
    public:
        float x;
        float y;

        __host__ __device__ Complex(float x, float y) {
            this -> x = x;
            this -> y = y;
        }

        __host__ __device__ float magnitude2(void) {
            return ((this -> x) * (this -> x) + (this -> y) * (this -> y));
        }
        __host__ __device__ Complex operator* (const Complex num) {
            return Complex((this -> x) * num.x - (this -> y) * num.y, (this -> y) * num.x + (this -> x) * num.y);
        }

        __host__ __device__ Complex operator+ (const Complex num) {
            return Complex(this -> x + num.x, this -> y + num.y);
        }
};

#endif