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
import graph.node.Results;
import math.Tensor;
import math.blas.BLAS;
import org.jocl.CL;
import org.jocl.cl_mem;

import java.util.Arrays;
import java.util.Map;

import static org.jocl.blast.CLBlastTranspose.CLBlastTransposeNo;
import static org.jocl.blast.CLBlastTranspose.CLBlastTransposeYes;

/**
 * A <code>GEMM</code> graph.node represents a graph.node which applies a matrix multiplication
 * operation to two tensors.
 */
public class GEMM extends Operation {
    private final boolean aTranspose, bTranspose;
    private final Dimension lda, ldb, ldc;

    public GEMM(boolean aTranspose, boolean bTranspose, Node... children) {
        super(children);

        if (children.length != 2)
            throw new IllegalArgumentException("Can only compute the GEMM of two tensors at a time.");

        this.aTranspose = aTranspose;
        this.bTranspose = bTranspose;

        lda = aTranspose ? Dimension.M : Dimension.K;
        ldb = bTranspose ? Dimension.K : Dimension.N;
        ldc = Dimension.N;
    }

    private int[] extractDimensions(int[] aDimensions, int[] bDimensions) {
        if (aDimensions.length != 2 || bDimensions.length != 2)
            throw new IllegalArgumentException("Can only compute the GEMM of 2D Tensors. Consider using a batched GEMM.");

        int m, n, k;
        if (aTranspose) {
            k = aDimensions[0];
            m = aDimensions[1];
        } else {
            m = aDimensions[0];
            k = aDimensions[1];
        }

        if (bTranspose) {
            n = bDimensions[0];
            if (k != bDimensions[1])
                throw new IllegalArgumentException(String.format("Incompatible k-dimension: '%s' != '%s'.", k, bDimensions[1]));
        } else {
            n = bDimensions[1];
            if (k != bDimensions[0])
                throw new IllegalArgumentException(String.format("Incompatible k-dimension: '%s' != '%s'.", k, bDimensions[0]));
        }

        return new int[]{m, n, k};
    }

    @Override
    protected Tensor computeOutput(Tensor[] inputs) {
        int[] aDimensions = inputs[0].getDimensions();
        int[] bDimensions = inputs[1].getDimensions();

        int[] mnk = extractDimensions(aDimensions, bDimensions);
        int m = mnk[0], n = mnk[1], k = mnk[2];

        cl_mem aBuffer = inputs[0].getBuffer(CL.CL_MEM_READ_ONLY);
        cl_mem bBuffer = inputs[1].getBuffer(CL.CL_MEM_READ_ONLY);

        Tensor c = new Tensor.Builder(m, n).build();
        cl_mem cBuffer = c.allocateBuffer(CL.CL_MEM_READ_WRITE);

        int ldaLength = this.lda == Dimension.K ? k : m;
        int ldbLength = this.ldb == Dimension.N ? n : k;
        int ldcLength = this.ldc == Dimension.N ? n : this.ldc == Dimension.K ? k : m;

        // m * k dot k * n
        // lda = k
        // ldb = n
        // ldc = n
        BLAS.sgemm(aBuffer, bBuffer, cBuffer, aTranspose ? CLBlastTransposeYes : CLBlastTransposeNo, bTranspose ? CLBlastTransposeYes : CLBlastTransposeNo, m, n, k, ldaLength, ldbLength, ldcLength);

        c.readFromBuffer();

        return c;
    }

    @Override
    protected Map<Long, Tensor> computeGradients(Map<Long, Tensor> gradients, Tensor delta) {
        // the derivative is the transposed other matrix
        Tensor a = Results.getOutput(children[0]);
        Tensor b = Results.getOutput(children[1]);

       int[] aDimensions = a.getDimensions();
        int[] bDimensions = b.getDimensions();

        int[] mnk = extractDimensions(aDimensions, bDimensions);
        int m = mnk[0], n = mnk[1], k = mnk[2];

        // a = m * k
        // b = k * n
        // c = m * n

        // dA = m * n dot n * k
        // dA = m * k
        // dB = k * m dot m * n
        // dB = k * n

        // dA = dC * dBT
        // dB = dAT * dC

        cl_mem aBuffer = a.allocateBuffer(CL.CL_MEM_READ_ONLY);
        cl_mem bBuffer = b.allocateBuffer(CL.CL_MEM_READ_ONLY);
        cl_mem deltaBuffer = delta.allocateBuffer(CL.CL_MEM_READ_ONLY);

        Tensor dA = new Tensor.Builder(a.getDimensions()).build();
        Tensor dB = new Tensor.Builder(b.getDimensions()).build();

        cl_mem dABuffer = dA.allocateBuffer(CL.CL_MEM_READ_WRITE);
        cl_mem dBBuffer = dB.allocateBuffer(CL.CL_MEM_READ_WRITE);

        BLAS.sgemm(deltaBuffer, bBuffer, dABuffer, CLBlastTransposeNo, bTranspose ? CLBlastTransposeNo : CLBlastTransposeYes, m, k, n, n, n, k);

        dA.readFromBuffer();
        BLAS.sgemm(aBuffer, deltaBuffer, dBBuffer, aTranspose ? CLBlastTransposeNo : CLBlastTransposeYes, CLBlastTransposeNo, k, n, m, k, n, n);

        dB.readFromBuffer();

        dA.releaseBuffer();
        dB.releaseBuffer();
        delta.releaseBuffer();

        gradients.put(children[0].getID(), dA);
        gradients.put(children[1].getID(), dB);

        return gradients;
    }

    public enum Dimension {
        M, N, K
    }
}
