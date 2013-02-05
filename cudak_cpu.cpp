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

class Cuda {
    private:
        CUdevice device;
        CUcontext context;
        CUmodule module;
    public:
        CUfunction raytracing_f;
        CUdeviceptr in;
        CUdeviceptr out;
        void init(){
            printf("device initialization\n");
            CHECK( cuInit(0) );
            CHECK(cuDeviceGet(&device, 0) );
            CHECK(cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device) );
            CHECK(cuModuleLoad(&module, "__cudak_gpu.ptx") );
            CHECK(cuModuleGetFunction(&raytracing_f, module, "black_and_white") );
            printf("device initialization succeeded\n");
        }
};

void wxInit() {
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

    for (int i = 0; i < width*height*3; ++i) {
        out[i] = bitmap[i];
    }
    printf("dupa\n");
    image.SaveFile(toWxString(output_path));
}

void black_and_white(Cuda & cuda, int w, int h){
    void * args[] = {&cuda.in, &cuda.out, &w, &h};
    CHECK( cuLaunchKernel(cuda.raytracing_f, w, h, 1,    1, 1    , 1, 0, NULL, args, NULL) );
    cuCtxSynchronize();
}

int main(int argc, char * argv[]) {
    wxInit();
    Cuda cuda;
    cuda.init();

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("input-path,I", po::value<string> (), "set input file path")
        ("output-path,o", po::value<string> (), "Place the output into <file>")
        ("black-and-white,b", "make image black and white");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

//     "-I " << vm["input-path"].as<string>() << endl;
    string input_path = vm["input-path"].as<string>();
    string output_path = vm["output-path"].as<string>();
//return 0;
    wxImage image = getImage(input_path);
    int w = image.GetWidth();
    int h = image.GetHeight();
    CHECK(cuMemAlloc(&cuda.in, 3*w*h));
    CHECK(cuMemcpyHtoD(cuda.in, image.GetData(), 3*w*h));
    CHECK(cuMemAlloc(&cuda.out, 3*w*h));

    if(vm.count("black-and-white")){
        cout << "bw " << w << " " << h << endl;
        black_and_white(cuda, w, h);
    }
    
    unsigned char *out;// = new unsigned char[3 * w * h];
    CHECK( cuMemAllocHost((void**)(&out), 3 * w * h) );
    CHECK( cuMemcpyDtoH(out, cuda.out, 3 * w * h) );
    save_bitmap(out, output_path, w, h);
}
