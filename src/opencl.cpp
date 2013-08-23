#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <immintrin.h>
#include <stdlib.h>

#include "common.h"

using namespace std;

static const double frequency = 1.0e-9;
static const int flops_per_calc = 2;
static const int dim = 5;
static const int nmat = 4992;
static const int iters = 1000;
static const int wg_size = 16;

const char * get_error_string(cl_int err){
  switch(err){
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";

    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    default: return "Unknown OpenCL error";
  }
}

int main(){
  std::string infostr;
  long infonum;

  scalar_t *A = (scalar_t *)_mm_malloc(sizeof(scalar_t)*nmat*dim*dim, 64);
  scalar_t *B = (scalar_t *)_mm_malloc(sizeof(scalar_t)*nmat*dim*dim, 64);
  scalar_t *C = (scalar_t *)_mm_malloc(sizeof(scalar_t)*nmat*dim*dim, 64);

  try{
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

    // 3. Device Fission : equally sub-divide the device
    cl_device_partition_property props[] = {
      CL_DEVICE_PARTITION_EQUALLY,
      1,
      CL_PROPERTIES_LIST_END_EXT
    };
    vector<cl::Device> sdevices;
    devices[0].createSubDevices(props, &sdevices);
    cerr << "Sub-divided into " << sdevices.size() << " devices" << endl;
    devices[0] = sdevices[0];

    devices[0].getInfo(CL_DEVICE_NAME, &infostr);
    cout << "Device name: " << infostr << endl;
    devices[0].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &infonum);
    cout << "Global memory size: " << infonum/(1024.*1024.*1024.) << " GB" << endl;
    devices[0].getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &infonum);
    cout << "Global cache size: " << infonum/1024 << " KB" << endl;
    devices[0].getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &infonum);
    cout << "Global cacheline size: " << infonum << " B" << endl;
    devices[0].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &infonum);
    cout << "Local memory size: " << infonum/1024 << " KB" << endl;
    devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &infonum);
    cout << "Max compute units: " << infonum << endl;
    devices[0].getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, &infonum);
    cout << "Preferred vector width char: " << infonum << endl;

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    ifstream file("opencl/gemm.cl");
    assert(file.is_open());

    std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));
    cl::Program program(context, source);
    program.build(devices,"");

    cl::Buffer inA(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, nmat*dim*dim*sizeof(scalar_t), A);
    cl::Buffer inB(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, nmat*dim*dim*sizeof(scalar_t), B);
    cl::Buffer outC(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, nmat*dim*dim*sizeof(scalar_t), C);

    cl::Kernel kernel(program, "benchmark");
    kernel.setArg(0, inA);
    kernel.setArg(1, inB);
    kernel.setArg(2, outC);
    kernel.setArg(3, dim);
    kernel.setArg(4, dim);
    kernel.setArg(5, dim);
    kernel.setArg(6, nmat);
    kernel.setArg(7, iters);

    double time = 0;
    cl::Event event;

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(nmat), cl::NDRange(wg_size), NULL, &event);
    event.wait();

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    time += (end - start);

    cout << time * 1.e-9 << endl;
    cl::finish();
  }catch(cl::Error error){
    cout << "error: " << error.what() << ": " << get_error_string(error.err()) << endl;
    return EXIT_FAILURE;
  }

  return 0;
}
