/*
 * MIT License
 *
 * Copyright (c) 2020 Patrick Song
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package math.blas;

import org.jocl.*;
import org.jocl.blast.CLBlastLayout;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;
import static org.jocl.blast.CLBlast.CLBlastSgemm;

public class BLAS {
    private static final cl_context context;
    private static final cl_command_queue commandQueue;

    private static final List<cl_mem> buffers = new ArrayList<>();

    static {
        int ret;

        ret = GPU.setupCL();
        if (ret != CL_SUCCESS)
            System.exit(ret);

        context = GPU.getContext();
        commandQueue = GPU.getCommandQueue();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Releasing resources.");
            for (cl_mem buffer : buffers) {
                clReleaseMemObject(buffer);
            }

            GPU.releaseResources();
        }));
    }

    public static void sgemm(cl_mem aBuffer, cl_mem bBuffer, cl_mem cBuffer, int aTranspose, int bTranspose, int m, int n, int k, int lda, int ldb, int ldc) {
        CLBlastSgemm(CLBlastLayout.CLBlastLayoutRowMajor, aTranspose, bTranspose, m, n, k, 1.0f, aBuffer, 0, lda, bBuffer, 0, ldb, 1.0f, cBuffer, 0, ldc, commandQueue, null);
    }

    public static cl_mem allocate(long flags, float[] values) {
        int ret;

        int[] retPointer = new int[1];
        cl_mem buffer = clCreateBuffer(context, flags, values.length * Sizeof.cl_float, null, retPointer);
        ret = retPointer[0];
        if (ret != CL_SUCCESS)
            System.exit(ret);

        ret = clEnqueueWriteBuffer(commandQueue, buffer, true, 0, values.length * Sizeof.cl_float, Pointer.to(values), 0, null, null);
        if (ret != CL_SUCCESS)
            System.exit(ret);

        buffers.add(buffer);

        return buffer;
    }

    public static void releaseBuffer(cl_mem buffer) {
        buffers.remove(buffer);
        clReleaseMemObject(buffer);
    }

    public static float[] readBuffer(cl_mem buffer, int length) {
        float[] result = new float[length];
        clEnqueueReadBuffer(commandQueue, buffer, true, 0, length * Sizeof.cl_float, Pointer.to(result), 0, null, null);

        return result;
    }
}
