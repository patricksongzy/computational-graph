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

import math.Tensor;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class OperationsTest {

    @Test
    void addition() {
        Tensor t1 = new Tensor.Builder().setDimensions(1, 2).setValues(0, 1).build();
        Tensor t2 = new Tensor.Builder().setDimensions(3, 2).setValues(2, 3, 4, 5, 6, 7).build();

        Tensor expected = new Tensor.Builder().setDimensions(3, 2).setValues(2, 4, 4, 6, 6, 8).build();

        Assertions.assertThat(Operations.addition(t1, t2)).isEqualTo(expected);
    }

    @Test
    void multiplication() {
        Tensor t1 = new Tensor.Builder().setDimensions(1, 2).setValues(0, 1).build();
        Tensor t2 = new Tensor.Builder().setDimensions(3, 2).setValues(2, 3, 4, 5, 6, 7).build();

        Tensor expected = new Tensor.Builder().setDimensions(3, 2).setValues(0, 3, 0, 5, 0, 7).build();

        assertThat(Operations.multiplication(t1, t2)).isEqualTo(expected);
    }

    @Test
    void division() {
        Tensor t1 = new Tensor.Builder().setDimensions(2).setValues(0, 1).build();
        Tensor t2 = new Tensor.Builder().setDimensions(3, 2).setValues(2, 5, 3, 10, 6, 2).build();

        Tensor expected = new Tensor.Builder().setDimensions(3, 2).setValues(0, 0.2f, 0, 0.1f, 0, 0.5f).build();

        assertThat(Operations.division(t1, t2)).isEqualTo(expected);
    }

    @Test
    void subtraction() {
        Tensor t1 = new Tensor.Builder().setDimensions(2).setValues(5, 2).build();
        Tensor t2 = new Tensor.Builder().setDimensions(3, 2).setValues(2, 9, 8, 5, 2, 1).build();

        // 5 2     2 9
        // 5 2     8 5
        // 5 2     2 1
        Tensor expected = new Tensor.Builder().setDimensions(3, 2).setValues(3, -7, -3, -3, 3, 1).build();

        assertThat(Operations.subtraction(t1, t2)).isEqualTo(expected);
    }

    @Test
    void sum() {
        Tensor t1 = new Tensor.Builder().setDimensions(2, 2, 3).setValues(9, 7, 5, 5, 3, 5, 1, 3, 5, 2, 6, 9).build();

        Tensor expected = new Tensor.Builder().setDimensions(2, 1).setValues(30, 30).build();

        assertThat(Operations.sum(t1, 0, 2)).isEqualTo(expected);
    }

    @Test
    void sumVector() {
        Tensor t1 = new Tensor.Builder().setDimensions(2, 3).setValues(2, 7, 2, 8, 1, 9).build();

        Tensor expected = new Tensor.Builder().setDimensions(1, 3).setValues(10, 8, 11).build();

        assertThat(Operations.sum(t1, 0)).isEqualTo(expected);
    }
}
