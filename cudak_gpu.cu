


extern "C" {

    __global__ void black_and_white(unsigned char * in, unsigned char * out, int w, int h){
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = 3 * (y * w + x);
        int bw = (in[idx] + in[idx+1] + in[idx+2]) / 3;
        if(idx < 3 * w * h){
            out[idx++] = bw;
            out[idx++] = bw;
            out[idx++] = bw;
        }
    }

    __global__ void transform(unsigned char * in, unsigned char * out, int w, int h){
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int idx_start = 3 * (y * w + x);
        double a [2][2];
        a[0][0] = 1;
        a[0][1] = 0;
        a[1][0] = 0;
        a[1][1] = 1;
        double tx = 0;
        double ty = 0;
        int x_end = a[0][0] * x + a[0][1] * y + tx;
        int y_end = a[1][0] * x + a[1][1] * y + ty;
        int idx_end = 3 * (y_end * w + x_end);
        if(idx_end < 3 * w * h){
            out[idx_end++] = in[idx_start++];
            out[idx_end++] = in[idx_start++];
            out[idx_end++] = in[idx_start++];
        }
    }
}
