import os

from triton._utils import parse_list_string
from triton.backends.intel.driver import compile_module_from_src, COMPILATION_HELPER, ty_to_cpp, serialize_args

# ------------------------
# Utils
# ------------------------

COMPILATION_HELPER.inject_pytorch_dep()

# ------------------------
# Launcher
# ------------------------


def make_launcher(constants, signature, ids):  # pylint: disable=unused-argument

    def _extracted_type(ty):
        if ty[0] == "*" or ty == "none":
            return "PyObject*"
        if ty[0] == "[":
            if ty == "[]":
                return "[]"
            tys = parse_list_string(ty)
            val = ",".join(map(_extracted_type, tys))
            return f"[{val}]"
        return ty_to_cpp(ty)

    def format_of(ty):
        if ty == "void*":
            return "O"
        if ty[0] == "[":
            if ty == "[]":
                return "()"
            tys = parse_list_string(ty)
            val = "".join(map(format_of, tys))
            return f"({val})"
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty]

    signature = {k: v for k, v in signature.items() if v != "constexpr"}
    args_format = "".join([format_of(_extracted_type(ty)) for ty in signature.values()])
    fmt = "iiiOOOOOO" + args_format
    signature = ",".join(signature.values()).replace("[", "").replace("]", "")
    signature = list(filter(bool, signature.split(",")))
    signature = dict(enumerate(signature))
    args_list = ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ""

    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    arg_decls = ", ".join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    # generate glue code
    src = f"""
    #include <cstddef>
    #include <string>
    #include <iostream>
    #include <iomanip>
    #include <level_zero/ze_api.h>
    #include <sycl/sycl.hpp>
    #include <ATen/record_function.h>

    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <Python.h>
    #include <stdio.h>
    #include <numpy/arrayobject.h>

    static inline void gpuAssert(ze_result_t code, const char *file, int line)
    {{
      if (code != ZE_RESULT_SUCCESS)
      {{
         const char* prefix = "Triton Error [ZE]: ";
         std::string str = std::to_string(code);
         char err[1024] = {{0}};
         strcat(err, prefix);
         strcat(err, str.c_str());
         PyErr_SetString(PyExc_RuntimeError, err);
      }}
    }}

    #define ZE_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

    typedef struct _DevicePtrInfo {{
      void* dev_ptr;
      bool valid;
    }} DevicePtrInfo;

    static inline void checkDevicePointer(DevicePtrInfo *ptr_info, int idx, const sycl::queue &queue) {{
      if (!ptr_info->dev_ptr || !ptr_info->valid) {{
        return;
      }}
      auto context = queue.get_context();
      auto handle = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
      ze_memory_allocation_properties_t prop;
      prop.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
      prop.pNext = nullptr;
      ze_device_handle_t device;
      auto res = zeMemGetAllocProperties((ze_context_handle_t)handle, ptr_info->dev_ptr, &prop, &device);
      if (res != ZE_RESULT_SUCCESS) {{
        PyErr_Format(PyExc_ValueError,
                     "Cannot get memory properties for pointer argument (at %d, err=%d)", idx, res);
        ptr_info->valid = false;
      }} else if (prop.type != ZE_MEMORY_TYPE_DEVICE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) doesn't reference XPU device memory (cpu tensor?)", idx);
        ptr_info->valid = false;
      }}
    }}

    static inline DevicePtrInfo getPointer(PyObject *obj, int idx, const sycl::queue &queue) {{
      DevicePtrInfo ptr_info;
      ptr_info.dev_ptr = 0;
      ptr_info.valid = true;
      if (PyLong_Check(obj)) {{
        ptr_info.dev_ptr = PyLong_AsVoidPtr(obj);
        checkDevicePointer(&ptr_info, idx, queue);
        return ptr_info;
      }}
      if (obj == Py_None) {{
        // valid nullptr
        return ptr_info;
      }}
      PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
      if(ptr){{
        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(ptr);
        if (!PyLong_Check(ret)) {{
          PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
          ptr_info.valid = false;
          return ptr_info;
        }}
        ptr_info.dev_ptr = PyLong_AsVoidPtr(ret);
        if(!ptr_info.dev_ptr) {{
          return ptr_info;
        }}
        checkDevicePointer(&ptr_info, idx, queue);
        Py_DECREF(ret);  // Thanks ChatGPT!
        return ptr_info;
      }}
      PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
      ptr_info.valid = false;
      return ptr_info;
    }}
// start sycl
  template <class T>
  static inline void set_scalar_arg(sycl::handler &cgh, int index, const void *value) {{
    cgh.set_arg(index, *static_cast<const T *>(value));
  }}
  static void sycl_kernel_launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_warps, int threads_per_warp, int shared_memory, sycl::queue& stream, sycl::kernel& kernel_ptr {", " + arg_decls if len(arg_decls) > 0 else ""}) {{

    std::string kernel_name = kernel_ptr.get_info<sycl::info::kernel::function_name>();
    RECORD_FUNCTION("XPU Triton kernel:" + kernel_name, {{}});
    void *params[] = {{ {", ".join(f"&arg{i}" for i, ty in signature.items() if i not in constants and ty != "none")} }};
    uint32_t num_params = sizeof(params)/sizeof(params[0]);
    uint32_t expected_num_params = kernel_ptr.get_info<sycl::info::kernel::num_args>();
    size_t global_range_x = gridX*threads_per_warp*num_warps;
    size_t global_range_y = gridY;
    size_t global_range_z = gridZ;
    size_t local_range_x = num_warps*threads_per_warp;
    size_t local_range_y = 1;
    size_t local_range_z = 1;
    sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
    sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
    sycl::nd_range<3> parallel_work_size(global_range, local_range);
    if (shared_memory) {{
      expected_num_params -= 1;
    }}
    assert(num_params == expected_num_params && "number of kernel param not matched");
    // Submit the imported kernel.
    auto cgf = [&](sycl::handler &cgh) {{
      {" ".join(f"set_scalar_arg<{ty_to_cpp(item)}>(cgh, {idx}, params[{idx}]);" for idx, item in enumerate([signature[i] for i in signature if i not in constants and signature[i] != "none"]))}      if (shared_memory) {{
          using share_mem_t = sycl::local_accessor<int8_t, 1>;
          share_mem_t local_buffer = share_mem_t(shared_memory, cgh);
          cgh.set_arg(num_params, local_buffer);
          cgh.parallel_for(parallel_work_size, kernel_ptr);
      }} else {{
          cgh.parallel_for(parallel_work_size, kernel_ptr);
      }}
      }};
    auto event = stream.submit(cgf);
  }}
// end sycl
    static PyObject* launch(PyObject* self, PyObject* args) {{

      int gridX, gridY, gridZ;
      PyObject *launch_enter_hook = NULL;
      PyObject *launch_exit_hook = NULL;
      PyObject *kernel_metadata = NULL;
      PyObject *launch_metadata = NULL;
      PyObject *py_obj_stream;
      PyObject *py_kernel;

      {" ".join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
      if(!PyArg_ParseTuple(args, \"{fmt}\", &gridX, &gridY, &gridZ, &py_obj_stream, &py_kernel,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
        return NULL;
      }}

      // extract kernel metadata
      int num_warps     = PyLong_AsLong(PyObject_GetAttrString(kernel_metadata, "num_warps"));
      int num_ctas      = PyLong_AsLong(PyObject_GetAttrString(kernel_metadata, "num_ctas"));
      int shared_memory = PyLong_AsLong(PyObject_GetAttrString(kernel_metadata, "shared"));
      int threads_per_warp = PyLong_AsLong(PyObject_GetAttrString(kernel_metadata, "threads_per_warp"));

      // extract cluster dims
      PyObject *clusterDim =  PyObject_GetAttrString(kernel_metadata, "cluster_dims");
      if (!PyTuple_Check(kernel_metadata)) {{
        PyErr_SetString(PyExc_TypeError, "kernel_metadata.cluster_dims must be a tuple");
        return NULL;
      }}
      int clusterDimX   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 0));
      int clusterDimY   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 1));
      int clusterDimZ   = PyLong_AsLong(PyTuple_GetItem(clusterDim, 2));
      // extract launch metadata
      if (launch_enter_hook != Py_None){{
        PyObject* args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
        Py_DECREF(args);
        if (!ret)
          return NULL;
      }}

      void * pStream = PyLong_AsVoidPtr(py_obj_stream);
      //error check
      if(pStream == nullptr || py_kernel == nullptr) return NULL;

      sycl::queue stream = *(static_cast<sycl::queue*>(pStream));
      sycl::kernel* kernel_ptr = reinterpret_cast<sycl::kernel*>(PyCapsule_GetPointer(py_kernel, "kernel"));
      if(kernel_ptr == nullptr) return NULL;
      sycl::kernel kernel = *kernel_ptr;

      {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}, stream); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" or ty == "none" else "" for i, ty in signature.items()])};
      sycl_kernel_launch(gridX, gridY, gridZ, num_warps, threads_per_warp, shared_memory, stream, kernel {"," + ", ".join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" or ty == "none" else f"_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ""});

      if(launch_exit_hook != Py_None){{
        PyObject* args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
        Py_DECREF(args);
        if (!ret)
          return NULL;
      }}
      if (PyErr_Occurred()) {{
        return NULL;
      }}

      // return None
      Py_INCREF(Py_None);
      return Py_None;
    }}

    static PyMethodDef ModuleMethods[] = {{
      {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
      {{NULL, NULL, 0, NULL}} // sentinel
    }};

    static struct PyModuleDef ModuleDef = {{
      PyModuleDef_HEAD_INIT,
      \"__triton_launcher\",
      NULL, //documentation
      -1, //size
      ModuleMethods
    }};

    PyMODINIT_FUNC PyInit___triton_launcher(void) {{
      PyObject *m = PyModule_Create(&ModuleDef);
      if(m == NULL) {{
        return NULL;
      }}
      PyModule_AddFunctions(m, ModuleMethods);
      return m;
    }}
    """
    return src


class XPULauncher:

    def __init__(self, src, metadata):  # pylint: disable=unused-argument
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else {}
        self.constants = dict(constants.items())
        self.signature = dict(src.signature.items())
        src = make_launcher(self.constants, self.signature, ids)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        # Serialize KernelArguments for SPIR-V Runner
        serialize_kernel_args = os.getenv("TRITON_XPU_DUMP_SPIRV_KERNEL_ARGS", None)
        if serialize_kernel_args:
            serialize_args(args, self.constants, self.signature)
        self.launch(*args, **kwargs)
