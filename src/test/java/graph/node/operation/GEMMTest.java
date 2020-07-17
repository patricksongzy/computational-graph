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

import graph.exception.NodeComputationException;
import graph.node.Graph;
import graph.node.Results;
import graph.node.leaves.Constant;
import math.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class GEMMTest {

    /**
     * Tests the matrix multiplication on the GPU.
     */
    @Test
    void gemmTest() {
        Constant a = new Constant(new Tensor.Builder(2, 3).setValues(2, 1, 4, 0, 1, 1).build());
        Constant b = new Constant(new Tensor.Builder(3, 4).setValues(6, 3, -1, 0, 1, 1, 0, 4, -2, 5, 0, 2).build());

        GEMM c = new GEMM(false, false, a, b);

        Graph.compute(null, c);

        assertThat(Results.getOutput(c)).isEqualTo(new Tensor.Builder(2, 4).setValues(5, 27, -2, 12, -1, 6, 0, 6).build());
    }

    @Test
    void dimensionsTest() {
        Constant a = new Constant(new Tensor.Builder(2, 3).setValues(2, 1, 4, 0, 1, 1).build());
        Constant b = new Constant(new Tensor.Builder(3, 4).setValues(6, 3, -1, 0, 1, 1, 0, 4, -2, 5, 0, 2).build());

        GEMM c = new GEMM(true, false, a, b);
        GEMM d = new GEMM(false, true, a, b);
        GEMM e = new GEMM(true, true, a, b);

        assertThrows(NodeComputationException.class, () -> Graph.compute(null, c));
        assertThrows(NodeComputationException.class, () -> Graph.compute(null, d));
        assertThrows(NodeComputationException.class, () -> Graph.compute(null, e));
    }

    /**
     * Tests the matrix multiplication on the GPU, when A is transposed.
     */
    @Test
    void gemmTranposeTestA() {
        Constant a = new Constant(new Tensor.Builder(3, 2).setValues(2, 0, 1, 1, 4, 1).build());
        Constant b = new Constant(new Tensor.Builder(3, 4).setValues(6, 3, -1, 0, 1, 1, 0, 4, -2, 5, 0, 2).build());

        GEMM c = new GEMM(true, false, a, b);

        Graph.compute(null, c);

        assertThat(Results.getOutput(c)).isEqualTo(new Tensor.Builder(2, 4).setValues(5, 27, -2, 12, -1, 6, 0, 6).build());
    }

    /**
     * Tests the matrix multiplication on the GPU, when B is transposed.
     */
    @Test
    void gemmTranposeTestB() {
        Constant a = new Constant(new Tensor.Builder(2, 3).setValues(2, 1, 4, 0, 1, 1).build());
        Constant b = new Constant(new Tensor.Builder(4, 3).setValues(6, 1, -2, 3, 1, 5, -1, 0, 0, 0, 4, 2).build());

        GEMM c = new GEMM(false, true, a, b);

        Graph.compute(null, c);

        assertThat(Results.getOutput(c)).isEqualTo(new Tensor.Builder(2, 4).setValues(5, 27, -2, 12, -1, 6, 0, 6).build());
    }

    /**
     * Tests the matrix multiplication on the GPU, when A, and B are transposed.
     */
    @Test
    void gemmTranposeTestAB() {
        Constant a = new Constant(new Tensor.Builder(3, 2).setValues(2, 0, 1, 1, 4, 1).build());
        Constant b = new Constant(new Tensor.Builder(4, 3).setValues(6, 1, -2, 3, 1, 5, -1, 0, 0, 0, 4, 2).build());

        GEMM c = new GEMM(true, true, a, b);

        Graph.compute(null, c);

        assertThat(Results.getOutput(c)).isEqualTo(new Tensor.Builder(2, 4).setValues(5, 27, -2, 12, -1, 6, 0, 6).build());
    }

    /**
     * Tests the gradient calculation for matrix multiplication on the GPU.
     */
    @Test
    void gemmGradientTest() {
        Constant a = new Constant(new Tensor.Builder(2, 3).setValues(2, 1, 4, 0, 1, 1).build());
        Constant b = new Constant(new Tensor.Builder(3, 4).setValues(6, 3, -1, 0, 1, 1, 0, 4, -2, 5, 0, 2).build());

        GEMM c = new GEMM(false, false, a, b);

        Graph.compute(null, c);
        Graph.gradient();

        assertThat(Results.getGradient(a)).isEqualTo(new Tensor.Builder(2, 3).setValues(8, 6, 5, 8, 6, 5).build());
        assertThat(Results.getGradient(b)).isEqualTo(new Tensor.Builder(3, 4).setValues(2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5).build());
    }

    /**
     * Tests the gradient calculation for matrix multiplication on the GPU, when A is transposed.
     */
    @Test
    void gemmGradientTransposeTestA() {
        Constant a = new Constant(new Tensor.Builder(3, 2).setValues(2, 0, 1, 1, 4, 1).build());
        Constant b = new Constant(new Tensor.Builder(3, 4).setValues(6, 3, -1, 0, 1, 1, 0, 4, -2, 5, 0, 2).build());

        GEMM c = new GEMM(true, false, a, b);

        Graph.compute(null, c);
        Graph.gradient();

        assertThat(Results.getGradient(a)).isEqualTo(new Tensor.Builder(3, 2).setValues(8, 8, 6, 6, 5, 5).build());
        assertThat(Results.getGradient(b)).isEqualTo(new Tensor.Builder(3, 4).setValues(2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5).build());
    }

    /**
     * Tests the gradient calculation for matrix multiplication on the GPU, when B is transposed.
     */
    @Test
    void gemmGradientTransposeTestB() {
        Constant a = new Constant(new Tensor.Builder(2, 3).setValues(2, 1, 4, 0, 1, 1).build());
        Constant b = new Constant(new Tensor.Builder(4, 3).setValues(6, 1, -2, 3, 1, 5, -1, 0, 0, 0, 4, 2).build());

        GEMM c = new GEMM(false, true, a, b);

        Graph.compute(null, c);
        Graph.gradient();

        assertThat(Results.getGradient(a)).isEqualTo(new Tensor.Builder(2, 3).setValues(8, 6, 5, 8, 6, 5).build());
        assertThat(Results.getGradient(b)).isEqualTo(new Tensor.Builder(4, 3).setValues(2, 2, 5, 2, 2, 5, 2, 2, 5, 2, 2, 5).build());
    }

    /**
     * Tests the gradient calculation for matrix multiplication on the GPU, when A, and B are transposed.
     */
    @Test
    void gemmGradientTransposeTestAB() {
        Constant a = new Constant(new Tensor.Builder(3, 2).setValues(2, 0, 1, 1, 4, 1).build());
        Constant b = new Constant(new Tensor.Builder(4, 3).setValues(6, 1, -2, 3, 1, 5, -1, 0, 0, 0, 4, 2).build());

        GEMM c = new GEMM(true, true, a, b);

        Graph.compute(null, c);
        Graph.gradient();

        assertThat(Results.getGradient(a)).isEqualTo(new Tensor.Builder(3, 2).setValues(8, 8, 6, 6, 5, 5).build());
        assertThat(Results.getGradient(b)).isEqualTo(new Tensor.Builder(4, 3).setValues(2, 2, 5, 2, 2, 5, 2, 2, 5, 2, 2, 5).build());
    }

    /**
     * Tears down after each test. All graphs are cleared, in order to ensure that nodes which are not used do not affect results.
     */
    @AfterEach
    void teardown() {
        Graph.clearAll();
    }
}