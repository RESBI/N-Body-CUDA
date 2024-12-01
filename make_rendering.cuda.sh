nvcc rendering.cu -o rendering.cuda.out -O5 -prec-sqrt=true -prec-div=true -Xcompiler "-openmp:experimental"
