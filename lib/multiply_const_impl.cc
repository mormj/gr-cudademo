/* -*- c++ -*- */
/*
 * Copyright 2021 Josh.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cudaerror.h"
#include "multiply_const_impl.h"
#include <gnuradio/io_signature.h>

extern void get_block_and_grid(int* minGrid, int* minBlock);
extern void exec_multiply_const_kernel(const float* in,
                                       float* out,
                                       float k,
                                       int grid_size,
                                       int block_size,
                                       size_t n,
                                       cudaStream_t stream);

namespace gr {
namespace cudademo {

using input_type = float;
using output_type = float;
multiply_const::sptr multiply_const::make(float k)
{
    return gnuradio::make_block_sptr<multiply_const_impl>(k);
}


/*
 * The private constructor
 */
multiply_const_impl::multiply_const_impl(float k)
    : gr::sync_block("multiply_const",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type))),
      d_k(k)
{
    get_block_and_grid(&d_min_grid_size, &d_block_size);

    check_cuda_errors(cudaStreamCreate(&d_stream));
    check_cuda_errors(cudaMalloc((void**)&d_dev_in, d_max_buffer_size));
    check_cuda_errors(cudaMalloc((void**)&d_dev_out, d_max_buffer_size));
}

/*
 * Our virtual destructor.
 */
multiply_const_impl::~multiply_const_impl() {}

int multiply_const_impl::work(int noutput_items,
                              gr_vector_const_void_star& input_items,
                              gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    check_cuda_errors(cudaMemcpyAsync(
        d_dev_in, in, noutput_items * sizeof(float), cudaMemcpyHostToDevice, d_stream));

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    exec_multiply_const_kernel(
        d_dev_in, d_dev_out, d_k, gridSize, d_block_size, noutput_items, d_stream);

    cudaMemcpyAsync(
        out, d_dev_out, noutput_items * sizeof(float), cudaMemcpyDeviceToHost, d_stream);

    cudaStreamSynchronize(d_stream);


    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace cudademo */
} /* namespace gr */
