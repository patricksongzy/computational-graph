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

import java.util.Scanner;

import static org.jocl.CL.*;

public class GPU {
    private static cl_context context;
    private static cl_command_queue commandQueue;

    static cl_context getContext() {
        return context;
    }

    static cl_command_queue getCommandQueue() {
        return commandQueue;
    }

    static void releaseResources() {
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    static int setupCL() {
        int ret;

        int[] numPlatforms = new int[1];
        int[] numDevices = new int[1];

        int platformIndex;
        int deviceIndex;

        cl_platform_id[] platforms;
        cl_device_id[] devices;

        ret = clGetPlatformIDs(0, null, numPlatforms);
        if (ret != CL_SUCCESS)
            return ret;

        platforms = new cl_platform_id[numPlatforms[0]];
        ret = clGetPlatformIDs(platforms.length, platforms, null);
        if (ret != CL_SUCCESS)
            return ret;

        System.out.println(String.format("There are %d platforms:", platforms.length));
        for (int i = 0; i < platforms.length; i++) {
            long[] parameterSize = new long[1];
            ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, null, parameterSize);
            if (ret != CL_SUCCESS)
                return ret;

            byte[] buffer = new byte[(int) parameterSize[0]];
            ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, buffer.length, Pointer.to(buffer), null);
            if (ret != CL_SUCCESS)
                return ret;

            // exclude the trailing \0
            String platformName = new String(buffer, 0, buffer.length - 1);

            System.out.println(String.format("%d: %s", i, platformName));
        }

        platformIndex = platforms.length == 1 ? 0 : getChoice("platform index", platforms.length - 1);
        System.out.println(String.format("Using platform %d.", platformIndex));

        ret = clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, 0, null, numDevices);
        if (ret != CL_SUCCESS)
            return ret;

        devices = new cl_device_id[numDevices[0]];
        ret = clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, devices.length, devices, null);
        if (ret != CL_SUCCESS)
            return ret;

        System.out.println(String.format("There are %d devices:", devices.length));
        for (int i = 0; i < devices.length; i++) {
            long[] parameterSize = new long[1];
            ret = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, null, parameterSize);
            if (ret != CL_SUCCESS)
                return ret;

            byte[] buffer = new byte[(int) parameterSize[0]];
            ret = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);
            if (ret != CL_SUCCESS)
                return ret;

            // exclude the trailing \0
            String deviceName = new String(buffer, 0, buffer.length - 1);

            System.out.println(String.format("%d: %s", i, deviceName));
        }

        deviceIndex = devices.length == 1 ? 0 : getChoice("device index", devices.length - 1);
        System.out.println(String.format("Using device %d.", deviceIndex));

        cl_context_properties properties = new cl_context_properties();
        properties.addProperty(CL_CONTEXT_PLATFORM, platforms[platformIndex]);

        int[] retPointer = new int[1];
        context = clCreateContext(properties, 1, new cl_device_id[]{devices[deviceIndex]}, null, null, retPointer);
        ret = retPointer[0];
        if (ret != CL_SUCCESS)
            return ret;

        commandQueue = clCreateCommandQueueWithProperties(context, devices[deviceIndex], null, retPointer);
        ret = retPointer[0];
        if (ret != CL_SUCCESS)
            clReleaseContext(context);

        return ret;
    }

    static int getChoice(String label, int maxChoice) {
        Scanner scanner = new Scanner(System.in);

        int next = -1;
        do {
            System.out.print(String.format("Please enter a %s in [%d, %d]: ", label, 0, maxChoice));
            if (scanner.hasNextInt()) {
                next = scanner.nextInt();
            } else {
                scanner.next();
                System.out.println("Input must be an integer.");
            }
        } while (next < 0 || next > maxChoice);

        return next;
    }
}
