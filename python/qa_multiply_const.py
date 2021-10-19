#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Josh.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks
try:
    from cudademo import multiply_const
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from cudademo import multiply_const

class qa_multiply_const(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        instance = multiply_const(1)

    def test_001_descriptive_test_name(self):
        nsamples = 10000

        k = 100.0
        input_data = list(range(nsamples))
        expected_data = [k*x for x in input_data]
        src = blocks.vector_source_f(input_data, False)
        mc = multiply_const(k)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, mc, snk)

        self.tb.run()


        self.assertEqual(snk.data(), expected_data)


if __name__ == '__main__':
    gr_unittest.run(qa_multiply_const)
