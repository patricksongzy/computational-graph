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

import static org.jocl.CL.*;

public class BLAS {
    private static cl_device_id device;

    private static cl_context context;
    private static cl_command_queue commandQueue;

    static {
         int ret;

         ret = GPU.setupCL();
         if (ret != CL_SUCCESS)
             System.exit(ret);

         device = GPU.getDevice();
         context = GPU.getContext();
         commandQueue = GPU.getCommandQueue();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Releasing resources.");
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
        }));
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

        return buffer;
    }
}
