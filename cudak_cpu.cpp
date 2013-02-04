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
namespace po = boost::program_options;

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

wxString toWxString(const string & s){
    return wxString(s.c_str(), wxConvUTF8);
}

wxImage getImage(const string & path){
    return wxImage (toWxString(path));
}

void save_bitmap(unsigned char * bitmap, const string & output_path, int width, int height){
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
    image.SaveFile(toWxString(output_path));
}

void run(const string & input_path, const string & output_path){
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
    
    wxImage image = getImage(input_path);
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

    save_bitmap(out, output_path, w, h);
}

int main(int argc, char * argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("input-path,I", po::value<string> (), "set input file path")
        ("output-path,o", po::value<string> (), "Place the output into <file>");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    cout << "-I " << vm["input-path"].as<string>() << endl;
    string input_path = vm["input-path"].as<string>();
    string output_path = vm["output-path"].as<string>();
    return 0;
    run(input_path, output_path);
}
