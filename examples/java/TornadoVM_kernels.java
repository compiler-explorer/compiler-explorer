// TornadoVM kernels: pick a "TornadoVM TC" compiler to see the CUDA, OpenCL or Metal code TC
// generates for every static void method over primitives and TornadoVM types.
// Options: --cc 90 (CUDA compute capability), --fp16 off, --fp64 off.
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

class Kernels {
    // A @Parallel loop: TC turns the loop into one GPU thread per iteration.
    static void saxpy(float alpha, FloatArray x, FloatArray y) {
        for (@Parallel int i = 0; i < y.getSize(); i++) {
            y.set(i, alpha * x.get(i) + y.get(i));
        }
    }

    // KernelContext: thread ids, local (shared) memory and barriers, as in CUDA or OpenCL.
    static void blockSum(KernelContext context, FloatArray in, FloatArray out) {
        float[] partial = context.allocateFloatLocalArray(256);
        int local = context.localIdx;
        partial[local] = in.get(context.globalIdx);
        for (int stride = context.localGroupSizeX / 2; stride > 0; stride /= 2) {
            context.localBarrier();
            if (local < stride) {
                partial[local] += partial[local + stride];
            }
        }
        if (local == 0) {
            out.set(context.groupIdx, partial[0]);
        }
    }
}
