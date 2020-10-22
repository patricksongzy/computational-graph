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

package math;

import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

class TensorTest {

    @Test
    void broadcast() {
        Tensor t1 = Tensor.zeros(4, 3, 1);
        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        Tensor t2 = Tensor.zeros(1, 2);
        for (int i = 0; i < t2.getLength(); i++) {
            t2.getValues()[i] = i;
        }

        Tensor[] results = Tensor.broadcast(t1, t2);
        Tensor[] expected = new Tensor[2];
        expected[0] = new Tensor.Builder().setDimensions(4, 3, 2)
                .setValues(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11).build();
        expected[1] = new Tensor.Builder().setDimensions(4, 3, 2)
                .setValues(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1).build();

        assertThat(results).isEqualTo(expected);
    }

    @Test
    void unbroadcast() {
        Tensor t1 = new Tensor.Builder().setDimensions(3, 1, 2).setValues(1, 2, 2, 3, 3, 4).build();
        Tensor t2 = new Tensor.Builder().setDimensions(3, 3, 1).setValues(1, 2, 3, 5, 2, 3, 8, 7, 9).build();

        Tensor t1Broadcast = Tensor.broadcast(t1, t2)[0];
        Tensor expected = new Tensor.Builder().setDimensions(3, 1, 2).setValues(3, 6, 6, 9, 9, 12).build();

        assertThat(Tensor.unbroadcast(t1Broadcast, t1.getDimensions())).isEqualTo(expected);
    }

    @Test
    void getFlattenedIndex() {
        assertThat(Tensor.getFlattenedIndex(new int[]{3, 2}, new int[]{0, 0, 0, 2, 1})).isEqualTo(5);
    }

    @Test
    void broadcastFailed() {
        Tensor t1 = Tensor.zeros(3, 2);
        Tensor t2 = Tensor.zeros(3, 5);

        assertThatIllegalArgumentException().isThrownBy(() -> Tensor.broadcast(t1, t2));
    }

    @Test
    void broadcastMulti() {
        Tensor t1 = Tensor.zeros(4, 3, 1);
        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        Tensor t2 = Tensor.zeros(1, 2);
        for (int i = 0; i < t2.getLength(); i++) {
            t2.getValues()[i] = i;
        }

        Tensor t3 = Tensor.zeros(3, 2);
        for (int i = 0; i < t3.getLength(); i++) {
            t3.getValues()[i] = i;
        }

        Tensor[] results = Tensor.broadcast(t1, t2, t3);
        Tensor[] expected = new Tensor[3];

        expected[0] =
                new Tensor.Builder().setDimensions(4, 3, 2).setValues(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11).build();
        expected[1] = new Tensor.Builder().setDimensions(4, 3, 2).setValues(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1).build();
        expected[2] = new Tensor.Builder().setDimensions(4, 3, 2).setValues(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5).build();

        assertThat(results).isEqualTo(expected);
    }

    @Test
    void dimensionsMatch() {
        Tensor t1 = Tensor.zeros(3, 8);
        Tensor t2 = Tensor.zeros(3, 8);

        assertThat(Tensor.isDimensionsMismatch(t1, t2)).isFalse();
    }

    @Test
    void dimensionsMismatch() {
        Tensor t1 = Tensor.zeros(3, 2, 1);
        Tensor t2 = Tensor.zeros(1, 5);

        assertThat(Tensor.isDimensionsMismatch(t1, t2)).isTrue();
    }

    @Test
    void dimensionsMismatchLength() {
        Tensor t1 = Tensor.zeros(3, 3, 3);
        Tensor t2 = Tensor.zeros(3, 3);

        assertThat(Tensor.isDimensionsMismatch(t1, t2)).isTrue();

    }

    @Test
    void get() {
        Tensor t1 = Tensor.zeros(3, 2);

        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        assertThat(t1.get(1, 0)).isEqualTo(2);
    }

    @Test
    void getAbsolute() {
        Tensor t1 = Tensor.zeros(3, 2);

        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        assertThat(t1.get(2)).isEqualTo(2);
    }

    @Test
    void getBroadcastValue() {
        Tensor t1 = Tensor.zeros(5, 3);

        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        assertThat(t1.getBroadcastedValue(new int[]{1, 5, 3}, new int[]{1, 2, 2})).isEqualTo(8);
    }

    @Test
    void set() {
        Tensor t1 = Tensor.zeros(3, 2);
        t1.set(5, 0, 1);

        assertThat(t1.get(0, 1)).isEqualTo(5);
    }

    @Test
    void testToString() {
        Tensor t1 = Tensor.zeros(2, 2, 2, 2);

        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(
                Objects.requireNonNull(getClass().getClassLoader().getResourceAsStream("tensor-string-result.txt"))));

        String expected = br.lines().collect(Collectors.joining("\n"));

        assertThat(t1.toString()).isEqualTo(expected);
    }

    @Test
    void equalsTest() {
        Tensor t1 = new Tensor.Builder().setDimensions(3, 2).setValues(0, 5, 8, 2, 9, 6).build();
        Tensor t2 = new Tensor.Builder().setDimensions(3, 2).setValues(0, 5, 8, 2, 9, 6).build();
        Tensor t3 = new Tensor.Builder().setDimensions(2, 3).setValues(0, 5, 8, 2, 9, 6).build();
        Tensor t4 = new Tensor.Builder().setDimensions(3, 2).setValues(1, 5, 8, 2, 9, 6).build();

        assertThat(t1.equals(t2)).isEqualTo(true);
        assertThat(t1.equals(t3)).isEqualTo(false);
        assertThat(t1.equals(t4)).isEqualTo(false);
        assertThat(t3.equals(t4)).isEqualTo(false);
    }

    @Test
    void hashCodeTest() {
        Tensor t1 = new Tensor.Builder().setDimensions(3, 2).setValues(0, 5, 8, 2, 9, 6).build();
        Tensor t2 = new Tensor.Builder().setDimensions(3, 2).setValues(0, 5, 8, 2, 9, 6).build();
        Tensor t3 = new Tensor.Builder().setDimensions(2, 3).setValues(0, 5, 8, 2, 9, 6).build();
        Tensor t4 = new Tensor.Builder().setDimensions(3, 2).setValues(1, 5, 8, 2, 9, 6).build();

        Set<Integer> codes = new HashSet<>();
        codes.add(t1.hashCode());
        codes.add(t3.hashCode());
        codes.add(t4.hashCode());

        assertThat(codes.size()).isEqualTo(3);

        assertThat(t1.hashCode()).isEqualTo(t2.hashCode());
    }

    @Test
    void zeros() {
        int[] dimensions = new int[]{2, 1, 3, 5, 8, 2};
        int product = 1;
        for (int dimension : dimensions) {
            product *= dimension;
        }

        float[] values = new float[product];

        Tensor t1 = Tensor.zeros(dimensions);
        assertThat(t1.getDimensions()).isEqualTo(dimensions);
        assertThat(t1.getValues()).isEqualTo(values);
    }
}
