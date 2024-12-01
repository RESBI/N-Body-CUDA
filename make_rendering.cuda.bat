nvcc rendering.cu -o rendering.cuda.exe -O5 -prec-sqrt=true -prec-div=true -Xcompiler "-openmp:experimental"
