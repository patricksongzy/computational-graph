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

package graph.node.operation;

import graph.node.Node;
import graph.node.Operation;
import math.Tensor;
import math.blas.BLAS;
import org.jocl.CL;
import org.jocl.blast.CLBlastTranspose;
import org.jocl.cl_mem;

import java.util.Map;

/**
 * A <code>GEMM</code> graph.node represents a graph.node which applies a matrix multiplication
 * operation to two tensors.
 */
public class GEMM extends Operation {
    private int aTranspose, bTranspose;

    public GEMM(int aTranspose, int bTranspose, Node... children) {
        super(children);

        this.aTranspose = aTranspose;
        this.bTranspose = bTranspose;
    }

    @Override
    protected Tensor computeOutput(Tensor[] inputs) {
        if (inputs.length != 2)
            throw new IllegalArgumentException("Can only compute the GEMM of two tensors at a time.");

        int[] mk = inputs[0].getDimensions();
        int[] kn = inputs[1].getDimensions();

        if (mk.length != 2 || kn.length != 2)
            throw new IllegalArgumentException("Can only compute the GEMM of 2D Tensors. Consider using a batched GEMM.");

        if (mk[1] != kn[0])
            throw new IllegalArgumentException(String.format("Incompatible k-dimension: '%s' != '%s'.", mk[1], kn[0]));

        cl_mem aBuffer = inputs[0].allocateBuffer(CL.CL_MEM_READ_ONLY);
        cl_mem bBuffer = inputs[1].allocateBuffer(CL.CL_MEM_READ_ONLY);

        Tensor c = new Tensor.Builder(mk[0], kn[1]).build();
        cl_mem cBuffer = c.allocateBuffer(CL.CL_MEM_READ_WRITE);

        int lda = aTranspose == CLBlastTranspose.CLBlastTransposeNo ? mk[1] : mk[0];
        int ldb = bTranspose == CLBlastTranspose.CLBlastTransposeNo ? kn[1] : kn[0];

        BLAS.sgemm(aBuffer, bBuffer, cBuffer, aTranspose, bTranspose, mk[0], kn[1], kn[0], lda, ldb, kn[1]);

        c.readFromBuffer();

        return c;
    }

    @Override
    protected Map<Long, Tensor> computeGradients(Map<Long, Tensor> gradients, Tensor delta) {
        return super.computeGradients(gradients, delta);
    }
}
