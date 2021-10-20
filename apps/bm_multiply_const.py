#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.9.0.0-git

from gnuradio import gr, blocks
import cudademo
import sys
import signal
from argparse import ArgumentParser
import time


class benchmark_multiply_const(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(
            self, "Benchmark Multiply Const", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        nsamples = int(args.samples)
        veclen = args.veclen
        num_blocks = args.nblocks
        bufsize = args.bufsize

        ##################################################
        # Blocks
        ##################################################
        blks = []
        for i in range(num_blocks):
            blks.append(
                cudademo.multiply_const(1.0)
            )
            blks[i].set_min_output_buffer(bufsize)

        src = blocks.null_source(
            gr.sizeof_float)
        src.set_min_output_buffer(bufsize)
        snk = blocks.null_sink(
            gr.sizeof_float)
        snk.set_min_output_buffer(bufsize)
        hd = blocks.head(
            gr.sizeof_float, nsamples)
        hd.set_min_output_buffer(bufsize)

        ##################################################
        # Connections
        ##################################################
        self.connect(hd, blks[0])
        self.connect(src, hd)

        for i in range(1, num_blocks):
            self.connect((blks[i-1], 0), (blks[i], 0))

        self.connect((blks[num_blocks-1], 0),
                     snk)


def main(top_block_cls=benchmark_multiply_const, options=None):

    parser = ArgumentParser(
        description='Run a flowgraph iterating over parameters for benchmarking')
    parser.add_argument(
        '--rt_prio', help='enable realtime scheduling', action='store_true')
    parser.add_argument('--samples', type=int, default=1e9)
    parser.add_argument('--veclen', type=int, default=1)
    parser.add_argument('--nblocks', type=int, default=1)
    parser.add_argument('--bufsize', type=int, default=8192)

    args = parser.parse_args()
    print(args)

    if args.rt_prio and gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

    tb = top_block_cls(args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("starting ...")
    startt = time.time()
    tb.start()

    tb.wait()
    endt = time.time()

    print(f'[PROFILE_TIME]{endt-startt}[PROFILE_TIME]')


if __name__ == '__main__':
    main()
