#include <string>
#include <vector>
#include <climits>
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <list>
#include <stack>
#include <deque>
#include <cstdio>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <wx/wx.h>
#include "cuda.h"
#include "common.h"
#include <boost/program_options.hpp>

using namespace std;

typedef vector<int> vi;
typedef long long ll;
typedef vector<ll> vll;
typedef vector<double> vd;
template<class T> void pr(T a, T b) { for(T i = a; i != b; ++i) cout << *i << " "; cout << endl; }

#define CHECK(expr)  \
do { \
    CUresult res = expr; \
    if(res!=CUDA_SUCCESS){printf("ASSERTION FAILED: line %d, res = %d\n", __LINE__, res); exit(1);} \
} while(0)

void init() {
    wxInitialize();
    wxInitAllImageHandlers();
}

wxImage getImage(const wchar_t * path ){
    return wxImage (wxString(path));
}

void save_bitmap(unsigned char * bitmap, const wxString path, int width, int height){
    wxImage image;
    image.Create(width, height, false);
    unsigned char * out = image.GetData();
    int outIdx = 0;

    int inIdx = 0;

    printf("dupa\n");
    for (int i = 0; i < width*height*3; ++i) {
        out[i] = bitmap[i];
    }
    printf("dupa\n");
    image.SaveFile(path);
}

int run(){
    init();
    printf("device initialization\n");

    CHECK( cuInit(0) );

    CUdevice cudaDevice;
    CUcontext cudaContext;
    CUmodule module;
    CUfunction raytracing_f;

    CHECK( cuDeviceGet(&cudaDevice, 0) );
    CHECK( cuCtxCreate(&cudaContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cudaDevice) );
    CHECK( cuModuleLoad(&module, "__cudak_gpu.ptx") );
    CHECK( cuModuleGetFunction(&raytracing_f, module, "black_and_white") );

    printf("data initialization\n");
    
    wxImage image = getImage(L"./mis.jpg");
    int w = image.GetWidth();
    int h = image.GetHeight();

    printf("w = %d, h = %d\n", w, h);

    CUdeviceptr dev_in;
    CHECK( cuMemAlloc(&dev_in, 3*w*h) );
    CHECK( cuMemcpyHtoD(dev_in, image.GetData(), 3*w*h) );

    CUdeviceptr dev_out;
    CHECK(cuMemAlloc(&dev_out, 3*w*h));

    void * args[] = {&dev_in, &dev_out, &w, &h};
    
    CHECK( cuLaunchKernel(raytracing_f, w, h, 1,    1, 1    , 1, 0, NULL, args, NULL) );
    cuCtxSynchronize();

    unsigned char *out;// = new unsigned char[3 * w * h];
    CHECK( cuMemAllocHost((void**)(&out), 3 * w * h) );
    CHECK( cuMemcpyDtoH(out, dev_out, 3 * w * h) );

    save_bitmap(out, L"./out.jpg", w, h);
}

int main() {
    run();
}
