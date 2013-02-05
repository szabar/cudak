


extern "C" {

    __global__ void black_and_white(float * in, float * out, int w, int h){
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = 3 * (y * w + x);
        float bw = (in[idx] + in[idx+1] + in[idx+2]) / 3;
        if(idx < 3 * w * h){
            out[idx++] = bw;
            out[idx++] = bw;
            out[idx++] = bw;
        }
    }
/*
    __device__ float normalize(float f){
        if(f > 1) return 1;
        if(f < 0) return 0;
        return f;
    }

    __global__ void contrast(float * in, float * out, int w, int h, int C, int B){
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        float Cf = (C * 1.0) / 255;
        float Bf = (B * 1.0) / 255;
        int idx = 3 * (y * w + x);
        if(idx < 3 * w * h){
            out[idx] = normalize(in[idx] - Cf / (Bf - Cf));idx++;
            out[idx] = normalize(in[idx] - Cf / (Bf - Cf));idx++;
            out[idx] = normalize(in[idx] - Cf / (Bf - Cf));idx++;
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
*/
}
