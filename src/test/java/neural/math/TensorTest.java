/*
 * MIT License
 *
 * Copyright (c) 2019 Patrick Song
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

package neural.math;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

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
        expected[0] = new Tensor.Builder(4, 3, 2)
            .setValues(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11)
            .build();
        expected[1] = new Tensor.Builder(4, 3, 2)
            .setValues(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
            .build();

        assertArrayEquals(expected, results);
    }

    @Test
    void broadcastFailed() {
        Tensor t1 = Tensor.zeros(3, 2);
        Tensor t2 = Tensor.zeros(3, 5);

        assertThrows(IllegalArgumentException.class, () -> Tensor.broadcast(t1, t2));
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

        expected[0] = new Tensor.Builder(4, 3, 2)
            .setValues(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11)
            .build();
        expected[1] = new Tensor.Builder(4, 3, 2)
            .setValues(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
            .build();
        expected[2] = new Tensor.Builder(4, 3, 2)
            .setValues(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5)
            .build();

        assertArrayEquals(expected, results);
    }

    @Test
    void dimensionsMatch() {
        Tensor t1 = Tensor.zeros(3, 8);
        Tensor t2 = Tensor.zeros(3, 8);

        assertFalse(Tensor.isDimensionsMismatch(t1, t2));
    }

    @Test
    void dimensionsMismatch() {
        Tensor t1 = Tensor.zeros(3, 2, 1);
        Tensor t2 = Tensor.zeros(1, 5);

        assertTrue(Tensor.isDimensionsMismatch(t1, t2));
    }

    @Test
    void dimensionsMismatchLength() {
        Tensor t1 = Tensor.zeros(3, 3, 3);
        Tensor t2 = Tensor.zeros(3, 3);

        assertTrue(Tensor.isDimensionsMismatch(t1, t2));

    }

    @Test
    void get() {
        Tensor t1 = Tensor.zeros(3, 2);

        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        assertEquals(2, t1.get(1, 0));
    }

    @Test
    void getAbsolute() {
        Tensor t1 = Tensor.zeros(3, 2);

        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        assertEquals(2, t1.get(2));
    }

    @Test
    void getBroadcast() {
        Tensor t1 = Tensor.zeros(5, 3);

        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        assertEquals(t1.getBroadcast(new int[]{1, 5, 3}, new int[]{1, 2, 2}), 8);
    }

    @Test
    void set() {
        Tensor t1 = Tensor.zeros(3, 2);
        t1.set(5, 0, 1);

        assertEquals(5, t1.get(0, 1));
    }

    @Test
    void testToString() {
        Tensor t1 = Tensor.zeros(2, 2, 2, 2);

        for (int i = 0; i < t1.getLength(); i++) {
            t1.getValues()[i] = i;
        }

        String expected =
            "<Tensor: shape=(2 x 2 x 2 x 2)>\n" +
                "{\n" +
                " [\n" +
                "  [\n" +
                "   [\n" +
                "    0.0, 1.0\n" +
                "    2.0, 3.0\n" +
                "   ]\n" +
                "   [\n" +
                "    4.0, 5.0\n" +
                "    6.0, 7.0\n" +
                "   ]\n" +
                "  ]\n" +
                "  [\n" +
                "   [\n" +
                "    8.0, 9.0\n" +
                "    10.0, 11.0\n" +
                "   ]\n" +
                "   [\n" +
                "    12.0, 13.0\n" +
                "    14.0, 15.0\n" +
                "   ]\n" +
                "  ]\n" +
                " ]\n" +
                "}";

        assertEquals(expected, t1.toString());
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
        assertArrayEquals(t1.getDimensions(), dimensions);
        assertArrayEquals(t1.getValues(), values);
    }
}