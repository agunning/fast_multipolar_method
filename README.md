# fast_multipolar_method

Hacky attempt to do (ML) attention in much less than quadratic time using fast multipolar methods

Unfortunately doesn't seem to yield that much improvement yet (64 is a lot of dimensions to do the relevant vector/tensor calculalions in, and trying to otpimise for efficiency just seems to make it want to brute force everything, making it just a slower regular attention)
