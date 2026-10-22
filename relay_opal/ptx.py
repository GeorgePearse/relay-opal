"""Hand-written PTX, loaded through the CUDA driver API with ctypes and launched on torch tensors.

    module = PtxModule(source)                      # JIT-assembled by the driver; an assembly failure raises with the
    fn = module.get("adamw")                        # ptxas text (the info and error logs) in the message
    module.launch(fn, grid, block, [param, grad, exp_avg, exp_avg_sq, lr, n])

Arguments: torch tensors go in as their data pointer, Python ints as int32, floats as float32, and ctypes instances
as they are (c_int64 for a 64-bit count, for example). Modules are cached by source hash. Nothing is checked before a
kernel runs: a wrong pointer or index corrupts memory or kills the process, which is the deal the agent gets.
"""

import ctypes
import hashlib

import torch

CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
LOG_BYTES = 1 << 16

_driver = None


class PtxError(RuntimeError):
    pass


def driver():
    """libcuda with the handful of entry points used here, the primary context made current on this thread."""
    global _driver
    if _driver is not None:
        return _driver
    if not torch.cuda.is_available():
        raise PtxError("PTX needs a CUDA device")
    lib = None
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        raise PtxError("libcuda.so not found")
    for fn in (
        "cuInit",
        "cuGetErrorString",
        "cuCtxGetCurrent",
        "cuCtxSetCurrent",
        "cuDevicePrimaryCtxRetain",
        "cuModuleLoadDataEx",
        "cuModuleGetFunction",
        "cuLaunchKernel",
        "cuModuleUnload",
    ):
        getattr(lib, fn).restype = ctypes.c_int
    torch.cuda.init()
    torch.zeros(1, device="cuda")  # the runtime creates and binds the primary context
    _check(lib, lib.cuInit(0), "cuInit")
    ctx = ctypes.c_void_p()
    _check(lib, lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    if not ctx.value:
        device = torch.cuda.current_device()
        _check(lib, lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device), "cuDevicePrimaryCtxRetain")
        _check(lib, lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    _driver = lib
    return lib


def _check(lib, result, what, log=""):
    if result != 0:
        text = ctypes.c_char_p()
        lib.cuGetErrorString(result, ctypes.byref(text))
        message = f"{what}: CUDA error {result} ({(text.value or b'?').decode()})"
        if log:
            message += "\n" + log
        raise PtxError(message)


class PtxModule:
    """One PTX module: assembled once per source text, kernels fetched by name, launched on torch's current stream."""

    _cache = {}

    def __new__(cls, source: str):
        key = hashlib.sha256(source.encode()).hexdigest()
        if key in cls._cache:
            return cls._cache[key]
        self = super().__new__(cls)
        self._init(source, key)
        cls._cache[key] = self
        return self

    def _init(self, source, key):
        lib = driver()
        self.source, self.key, self.functions = source, key, {}
        info, error = ctypes.create_string_buffer(LOG_BYTES), ctypes.create_string_buffer(LOG_BYTES)
        options = (ctypes.c_int * 4)(
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        )
        values = (ctypes.c_void_p * 4)(
            ctypes.cast(info, ctypes.c_void_p), LOG_BYTES, ctypes.cast(error, ctypes.c_void_p), LOG_BYTES
        )
        self.module = ctypes.c_void_p()
        image = ctypes.create_string_buffer(source.encode())
        result = lib.cuModuleLoadDataEx(ctypes.byref(self.module), image, 4, options, values)
        self.info_log = info.value.decode(errors="replace")
        self.error_log = error.value.decode(errors="replace")
        _check(lib, result, "cuModuleLoadDataEx (ptxas)", f"ptxas: {self.error_log}\n{self.info_log}".strip())

    def get(self, name: str):
        if name not in self.functions:
            fn = ctypes.c_void_p()
            result = driver().cuModuleGetFunction(ctypes.byref(fn), self.module, name.encode())
            _check(driver(), result, f"cuModuleGetFunction {name!r}")
            self.functions[name] = fn
        return self.functions[name]

    def launch(self, fn, grid, block, args, shared=0, stream=None):
        """cuLaunchKernel. grid and block are ints or (x, y, z); args: tensors, ints, floats or ctypes values."""
        if isinstance(fn, str):
            fn = self.get(fn)
        gx, gy, gz = _dims(grid)
        bx, by, bz = _dims(block)
        packed = [_pack(a) for a in args]
        pointers = (ctypes.c_void_p * len(packed))(*[ctypes.cast(ctypes.pointer(p), ctypes.c_void_p) for p in packed])
        handle = (stream if stream is not None else torch.cuda.current_stream()).cuda_stream
        result = driver().cuLaunchKernel(
            fn, gx, gy, gz, bx, by, bz, int(shared), ctypes.c_void_p(handle), pointers, None
        )
        _check(driver(), result, "cuLaunchKernel")

    def __repr__(self):
        return f"PtxModule({self.key[:12]}, {sorted(self.functions)})"


def _dims(d):
    if isinstance(d, int):
        return d, 1, 1
    d = tuple(int(x) for x in d) + (1, 1)
    return d[0], d[1], d[2]


def _pack(a):
    if isinstance(a, torch.Tensor):
        return ctypes.c_void_p(a.data_ptr())
    if isinstance(a, bool):
        return ctypes.c_int(int(a))
    if isinstance(a, int):
        return ctypes.c_int(a)
    if isinstance(a, float):
        return ctypes.c_float(a)
    if isinstance(a, ctypes._SimpleCData):
        return a
    raise TypeError(f"cannot pass {type(a).__name__} to a kernel; use a tensor, int, float or a ctypes value")


def grid_for(n, block=256):
    """A one-dimensional grid for a grid-stride loop over n elements, capped at 65535 blocks."""
    return min(65535, max(1, (int(n) + block - 1) // block))


_restore_requests = []


def request_restore(update: int):
    """Ask the training loop to restore the checkpoint taken at or before `update` (parameters, optimizer state,
    estimator) after the current update, keeping the current programs. A tool for a program that decides a path was
    unproductive; nothing checks that it is called wisely."""
    _restore_requests.append(int(update))


def take_restore_request():
    if not _restore_requests:
        return None
    update = _restore_requests[-1]
    _restore_requests.clear()
    return update
