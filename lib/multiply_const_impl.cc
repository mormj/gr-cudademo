/* -*- c++ -*- */
/*
 * Copyright 2021 Josh.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cudaerror.h"
#include "multiply_const_impl.h"
#include <gnuradio/io_signature.h>
#include <cuda_buffer/cuda_buffer.h>

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
                         1, 1 , sizeof(input_type), cuda_buffer::type),
                     gr::io_signature::make(
                         1, 1 , sizeof(output_type), cuda_buffer::type)),
      d_k(k)
{
    get_block_and_grid(&d_min_grid_size, &d_block_size);

    check_cuda_errors(cudaStreamCreate(&d_stream));
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

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    exec_multiply_const_kernel(
        in, out, d_k, gridSize, d_block_size, noutput_items, d_stream);

    cudaStreamSynchronize(d_stream);


    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace cudademo */
} /* namespace gr */
