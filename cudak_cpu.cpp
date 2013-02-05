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

const int BLACK_AND_WHITE = 0;
const int CONTRAST = 1;

class Cuda {
    private:
        CUdevice device;
        CUcontext context;
        CUmodule module;
    public:
        vector<CUfunction> t;
        CUdeviceptr bitmap;
        void init(){
            printf("device initialization\n");
            CHECK( cuInit(0) );
            CHECK(cuDeviceGet(&device, 0) );
            CHECK(cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device) );
            CHECK(cuModuleLoad(&module, "__cudak_gpu.ptx") );
            t.resize(2);
            CHECK(cuModuleGetFunction(&t[BLACK_AND_WHITE], module, "black_and_white"));
            CHECK(cuModuleGetFunction(&t[CONTRAST], module, "contrast"));
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

void save_bitmap(float * bitmap, const string & output_path, int width, int height){
    wxImage image;
    image.Create(width, height, false);
    unsigned char * out = image.GetData();
    int outIdx = 0;

    int inIdx = 0;

    for (int i = 0; i < width*height*3; ++i) {
        out[i] = bitmap[i] * 255;
    }
    printf("dupa\n");
    image.SaveFile(toWxString(output_path));
}

void black_and_white(Cuda & cuda, int w, int h){
    void * args[] = {&cuda.bitmap, &w, &h};
    CHECK(cuLaunchKernel(cuda.t[BLACK_AND_WHITE], w, h, 1,    1, 1    , 1, 0, NULL, args, NULL));
    cuCtxSynchronize();
}

void contrast(Cuda & cuda, int w, int h, int C, int B){
    void * args[] = {&cuda.bitmap, &w, &h, &C, &B};
    CHECK(cuLaunchKernel(cuda.t[CONTRAST], w, h, 1,    1, 1    , 1, 0, NULL, args, NULL));
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
        ("black-and-white,b", "make image black and white")
        ("contrast-black", po::value<int> (), "")
        ("contrast-white", po::value<int> (), "")
        ;
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
    CHECK(cuMemAlloc(&cuda.bitmap, sizeof(float) * 3*w*h));
    float * bitmap = new float[3 * w * h];
    for(int i = 0; i < 3*w*h; ++i){
        bitmap[i] = (image.GetData()[i] * (1.0)) / (255.0);
    }
    CHECK(cuMemcpyHtoD(cuda.bitmap, bitmap, sizeof(float) * 3*w*h));

    if(vm.count("black-and-white")){
        cout << "bw " << w << " " << h << endl;
        black_and_white(cuda, w, h);
    }
    if(vm.count("contrast-black") || vm.count("contrast-white")){
        cout << "abcabc" << vm.count("contrast-black") << endl;
        contrast(cuda, w, h, vm["contrast-black"].as<int>(), vm["contrast-white"].as<int>());
    }
    
    float *out;// = new unsigned char[3 * w * h];
    CHECK(cuMemAllocHost((void**)(&out), sizeof(float) * 3 * w * h));
    CHECK(cuMemcpyDtoH(out, cuda.bitmap, sizeof(float) * 3 * w * h));
    printf("dupa dupa\n");
    save_bitmap(out, output_path, w, h);
}
